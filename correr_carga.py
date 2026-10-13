"""Cinco altas concurrentes y, después, una tarea crítica que espera aprobación."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

import levantar

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
SALIDA = Path(__file__).resolve().parent / "trazas" / "carga.json"

CASOS = [
    (
        "novax-margen",
        "Con las reseñas del Nova X, decime el sentimiento neto y el margen porcentual de la ficha.",
    ),
    (
        "novax-precio",
        "¿Cuál es el precio de venta y el costo de reposición del Nova X?",
    ),
    (
        "novax-luis",
        "¿Qué sentimiento tiene la reseña de Luis sobre los Nova X? Incluí el sentimiento neto y el margen.",
    ),
    (
        "novax-umbral",
        "Con la ficha del Nova X, ¿el margen porcentual supera el 50%? Mostrá sentimiento neto y margen.",
    ),
    (
        "fuera-de-corpus",
        "¿Cuál es la garantía en dólares del televisor Bravia? Si no está en las reseñas, decilo.",
    ),
]

CRITICA = (
    "novax-descuento",
    "Publicá en el catálogo un descuento del Nova X. "
    "Antes calculá el sentimiento neto y el margen porcentual de la ficha.",
)


async def _esperar(cliente: httpx.AsyncClient, job_id: str, hasta: float) -> dict:
    limite = time.monotonic() + hasta
    while time.monotonic() < limite:
        response = await cliente.get(f"{API}/tasks/{job_id}")
        response.raise_for_status()
        estado = response.json()
        if estado["status"] in {"DONE", "FAILED", "AWAITING_APPROVAL"}:
            return estado
        await asyncio.sleep(3)
    raise TimeoutError(f"El job {job_id} no terminó a tiempo")


async def _alta(cliente: httpx.AsyncClient, nombre: str, consulta: str) -> dict:
    inicio = time.perf_counter()
    response = await cliente.post(
        f"{API}/tasks",
        json={"consulta": consulta, "thread_id": nombre},
    )
    response.raise_for_status()
    aceptada = response.json()
    aceptada["alta_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    aceptada["nombre"] = nombre
    print(f"{nombre} -> {aceptada['job_id']} en {aceptada['alta_ms']} ms", flush=True)
    return aceptada


async def main() -> None:
    print("Aseguro Ollama (y Phoenix si hace falta)...", flush=True)
    levantar.main()

    async with httpx.AsyncClient(timeout=30) as cliente:
        salud = await cliente.get(f"{API}/health")
        salud.raise_for_status()
        if salud.json().get("redis") != "ok":
            raise SystemExit("Redis no responde")

        altas = await asyncio.gather(*[_alta(cliente, nombre, consulta) for nombre, consulta in CASOS])
        estados = []
        for alta in altas:
            estado = await _esperar(cliente, alta["job_id"], 2400)
            print(f"  {alta['nombre']} {estado['status']}", flush=True)
            estados.append({"caso": alta["nombre"], "alta_ms": alta["alta_ms"], **estado})

        critica = await _alta(cliente, *CRITICA)
        pausa = await _esperar(cliente, critica["job_id"], 900)
        print(f"  pausa {pausa['status']}: {pausa.get('propuesta')}", flush=True)
        if pausa["status"] != "AWAITING_APPROVAL":
            raise SystemExit(f"La tarea crítica no se detuvo: {pausa['status']} {pausa.get('error')}")
        aprobado = await cliente.post(
            f"{API}/tasks/{critica['job_id']}/approve",
            json={"aprobado": True, "comentario": "ok para el catálogo"},
        )
        aprobado.raise_for_status()
        print(f"  approve -> {aprobado.status_code} {aprobado.json()['status']}", flush=True)
        cierre = await _esperar(cliente, critica["job_id"], 600)
        print(f"  critica {cierre['status']}", flush=True)
        hilo = await cliente.get(f"{API}/conversations/novax-margen")
        hilo.raise_for_status()

    informe = {
        "altas_ms": [item["alta_ms"] for item in estados],
        "resultados": estados,
        "critica": {"alta_ms": critica["alta_ms"], "pausa": pausa, "cierre": cierre},
        "conversacion_margen": hilo.json()["respuesta_final"],
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    fallidos = [item["caso"] for item in estados if item["status"] != "DONE"]
    if fallidos or cierre["status"] != "DONE":
        raise SystemExit(f"Corrida incompleta: {fallidos} critica={cierre['status']}")
    print(f"Listo. Informe en {SALIDA}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
