"""Levanta Ollama y Phoenix si no responden."""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.config import EMBED_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, PHOENIX_URL, TRAZAS

_DETACHED = 0
if os.name == "nt":
    _DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP


def _ok(url: str, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _esperar(url: str, segundos: float, etiqueta: str) -> None:
    limite = time.monotonic() + segundos
    while time.monotonic() < limite:
        if _ok(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"{etiqueta} no respondió en {url} tras {int(segundos)}s")


def _lanzar(comando: list[str], log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    subprocess.Popen(
        comando,
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=_DETACHED,
        close_fds=True,
    )


def _ollama_exe() -> str:
    encontrado = shutil.which("ollama")
    if encontrado:
        return encontrado
    candidato = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if candidato.exists():
        return str(candidato)
    raise RuntimeError("No encontré Ollama.")


def _modelos() -> list[str]:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
        data = json.load(response)
    return [item["name"] for item in data.get("models", [])]


def _instalado(pedido: str, nombres: list[str]) -> bool:
    if pedido in nombres:
        return True
    if ":" not in pedido:
        return any(nombre.split(":")[0] == pedido for nombre in nombres)
    return False


def _post(url: str, payload: dict, timeout: float) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def asegurar_ollama() -> None:
    if not _ok(f"{OLLAMA_BASE_URL}/api/tags"):
        print("Ollama no responde. Lo inicio.", flush=True)
        _lanzar([_ollama_exe(), "serve"], TRAZAS / "warmup-ollama.log")
        _esperar(f"{OLLAMA_BASE_URL}/api/tags", 40, "Ollama")
    else:
        print("Ollama ya está levantado.", flush=True)

    nombres = _modelos()
    for modelo in (OLLAMA_MODEL, EMBED_MODEL):
        if _instalado(modelo, nombres):
            continue
        print(f"Descargo {modelo}.", flush=True)
        subprocess.run([_ollama_exe(), "pull", modelo], check=True)

    print(f"Cargo {OLLAMA_MODEL} y {EMBED_MODEL}.", flush=True)
    _post(
        f"{OLLAMA_BASE_URL}/api/generate",
        {
            "model": OLLAMA_MODEL,
            "prompt": "ok",
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": "30m",
        },
        240,
    )
    _post(f"{OLLAMA_BASE_URL}/api/embed", {"model": EMBED_MODEL, "input": "ok"}, 60)


def asegurar_phoenix() -> None:
    if _ok(PHOENIX_URL):
        print("Phoenix ya está levantado.", flush=True)
        return
    print("Phoenix no responde. Lo inicio.", flush=True)
    raiz = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("PHOENIX_PORT", "6006")
    env.setdefault("PHOENIX_HOST", "127.0.0.1")
    env.setdefault("PHOENIX_GRPC_PORT", "4318")
    env.setdefault("PHOENIX_WORKING_DIR", str(raiz / ".phoenix"))
    _lanzar(
        [sys.executable, "-m", "phoenix.server.main", "serve"],
        TRAZAS / "warmup-phoenix.log",
        env,
    )
    try:
        _esperar(PHOENIX_URL, 60, "Phoenix")
    except RuntimeError:
        log = TRAZAS / "warmup-phoenix.log"
        detalle = log.read_text(encoding="utf-8", errors="replace")[-800:] if log.exists() else ""
        raise RuntimeError(f"Phoenix no levantó.\n{detalle}") from None


def main() -> None:
    asegurar_ollama()
    asegurar_phoenix()


if __name__ == "__main__":
    main()
