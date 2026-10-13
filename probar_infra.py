"""Redis, la pausa humana y el paso a FAILED, sin llamar al modelo."""

import asyncio
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from redis import asyncio as aioredis

from app.config import REDIS_URL
from app.graph import memoria_redis
from app.worker import clave, ejecutar, guardar, leer


class Estado(TypedDict, total=False):
    messages: list
    respuesta_final: str
    pasos: int
    es_critica: bool
    aprobado: bool | None


def _pausar(state: Estado) -> dict:
    decision = interrupt(
        {
            "accion": "publicar_ajuste_precio",
            "motivo": "prueba",
            "precio_actual": 10.0,
            "precio_propuesto": 9.0,
            "margen_porcentual": 50.0,
        }
    )
    if not isinstance(decision, dict) or not decision.get("aprobado"):
        raise RuntimeError("la prueba esperaba una aprobación")
    return {
        "respuesta_final": "publicado en la prueba",
        "pasos": 1,
        "es_critica": True,
        "aprobado": True,
    }


async def _fallar(state: Estado) -> dict:
    raise RuntimeError("fallo de prueba")


async def _job(redis, consulta: str) -> str:
    job_id = f"infra-{uuid.uuid4()}"
    await guardar(
        redis,
        job_id,
        {
            "job_id": job_id,
            "thread_id": job_id,
            "status": "PENDING",
            "fase": "nueva",
            "consulta": consulta,
            "respuesta": None,
            "pasos": None,
            "error": None,
            "es_critica": False,
            "propuesta": None,
            "aprobado": None,
            "comentario": "",
        },
    )
    return job_id


async def main() -> None:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await redis.ping()
    async with memoria_redis(REDIS_URL) as checkpointer:
        pausa = StateGraph(Estado)
        pausa.add_node("aprobacion", _pausar)
        pausa.add_edge(START, "aprobacion")
        pausa.add_edge("aprobacion", END)
        grafo_pausa = pausa.compile(checkpointer=checkpointer)

        fallo = StateGraph(Estado)
        fallo.add_node("boom", _fallar)
        fallo.add_edge(START, "boom")
        fallo.add_edge("boom", END)
        grafo_fallo = fallo.compile(checkpointer=checkpointer)

        job_pausa = await _job(redis, "publicar precio")
        await ejecutar(grafo_pausa, redis, job_pausa)
        detenido = await leer(redis, job_pausa)
        if detenido["status"] != "AWAITING_APPROVAL":
            raise SystemExit(f"no pausó: {detenido}")
        checkpoint = await grafo_pausa.aget_state({"configurable": {"thread_id": job_pausa}})
        if not checkpoint.tasks and not getattr(checkpoint, "interrupts", ()):
            raise SystemExit("el checkpointer no guardó la interrupción")

        detenido["fase"] = "reanudar"
        detenido["status"] = "RUNNING"
        detenido["aprobado"] = True
        detenido["comentario"] = "ok"
        await guardar(redis, job_pausa, detenido)
        await ejecutar(grafo_pausa, redis, job_pausa)
        cerrado = await leer(redis, job_pausa)
        if cerrado["status"] != "DONE" or "publicado" not in (cerrado.get("respuesta") or ""):
            raise SystemExit(f"no reanudó: {cerrado}")

        job_fallo = await _job(redis, "esto falla")
        await ejecutar(grafo_fallo, redis, job_fallo)
        roto = await leer(redis, job_fallo)
        if roto["status"] != "FAILED" or "RuntimeError" not in (roto.get("error") or ""):
            raise SystemExit(f"no marcó FAILED: {roto}")

        await redis.delete(clave(job_pausa), clave(job_fallo))
    await redis.aclose()
    print("Redis, HITL y FAILED: ok")


if __name__ == "__main__":
    asyncio.run(main())
