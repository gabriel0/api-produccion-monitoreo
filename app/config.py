import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAZAS = ROOT / "trazas"

CORPUS_PATH = DATA / "corpus.json"
VECTORES_PATH = DATA / "vectores.json"
AJUSTES_PATH = DATA / "ajustes.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "2m")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://127.0.0.1:6006").rstrip("/")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT", "pre-entrega7")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

MAX_PASOS = int(os.getenv("MAX_PASOS", "4"))
MAX_VISITAS = int(os.getenv("MAX_VISITAS", "2"))
TOP_K = int(os.getenv("TOP_K", "4"))
VERSION_PROMPT = os.getenv("PROMPT_VERSION", "pre-entrega7-v1")
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "600"))

# Tarifas de referencia para que Phoenix calcule costo por tokens.
# Con Gemini usá valores cercanos a la tabla pública del modelo.
_default_in = "0.10" if LLM_PROVIDER == "gemini" else "2"
_default_out = "0.40" if LLM_PROVIDER == "gemini" else "8"
COSTO_ENTRADA_POR_MILLON = float(os.getenv("COSTO_ENTRADA_POR_MILLON", _default_in))
COSTO_SALIDA_POR_MILLON = float(os.getenv("COSTO_SALIDA_POR_MILLON", _default_out))

if LLM_PROVIDER not in {"ollama", "gemini"}:
    raise RuntimeError(f"LLM_PROVIDER inválido: {LLM_PROVIDER!r}. Usá ollama o gemini.")

QUEUE_NAME = "preentrega7:tareas"
JOB_PREFIX = "preentrega7:job:"

PRECIO_FICHA = 89990.0
COSTO_FICHA = 40000.0
