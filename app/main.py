"""API asíncrona del orquestador. La instrumentación corre antes de LangChain."""

from dotenv import load_dotenv

load_dotenv()

from app.observability import iniciar

iniciar()

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from redis import asyncio as aioredis

from app.agents.runtime import texto
from app.agents.vector_store import cerrar_http
from app.config import API_HOST, API_PORT, QUEUE_NAME, REDIS_URL
from app.graph import build_graph, memoria_redis
from app.observability import registrar_tarifa
from app.schemas import (
    AprobacionAceptada,
    AprobacionEntrada,
    Conversacion,
    EstadoJob,
    MensajeConversacion,
    Salud,
    TareaAceptada,
    TareaEntrada,
)
from app.worker import consumir, guardar, leer

log = logging.getLogger("preentrega7")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis_worker = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=None,
        socket_connect_timeout=5,
    )
    await app.state.redis.ping()
    await registrar_tarifa()
    async with memoria_redis(REDIS_URL) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        tarea = asyncio.create_task(consumir(app))
        yield
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            pass
    await app.state.redis_worker.aclose()
    await app.state.redis.aclose()
    await cerrar_http()


app = FastAPI(
    title="Sistema de inteligencia de producto",
    version="1.0.0",
    description=(
        "Orquestador del Nova X con cola Redis, checkpoints en Redis, "
        "trazas en Phoenix y aprobación humana antes de publicar un precio."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=Salud, tags=["sistema"])
async def health() -> Salud:
    try:
        await app.state.redis.ping()
        redis_ok = "ok"
    except Exception:
        redis_ok = "error"
    return Salud(api="ok", redis=redis_ok)


@app.post("/tasks", response_model=TareaAceptada, status_code=202, tags=["tareas"])
async def crear_tarea(entrada: TareaEntrada) -> TareaAceptada:
    job_id = str(uuid.uuid4())
    thread_id = entrada.thread_id or str(uuid.uuid4())
    estado = {
        "job_id": job_id,
        "thread_id": thread_id,
        "status": "PENDING",
        "fase": "nueva",
        "consulta": entrada.consulta,
        "respuesta": None,
        "pasos": None,
        "error": None,
        "es_critica": False,
        "propuesta": None,
        "aprobado": None,
        "comentario": "",
    }
    await guardar(app.state.redis, job_id, estado)
    await app.state.redis.rpush(QUEUE_NAME, job_id)
    return TareaAceptada(job_id=job_id, thread_id=thread_id, status="PENDING")


@app.get("/tasks/{job_id}", response_model=EstadoJob, tags=["tareas"])
async def estado_tarea(job_id: str) -> EstadoJob:
    estado = await leer(app.state.redis, job_id)
    if estado is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return EstadoJob.model_validate(estado)


@app.post(
    "/tasks/{job_id}/approve",
    response_model=AprobacionAceptada,
    status_code=202,
    tags=["tareas"],
)
async def aprobar_tarea(job_id: str, entrada: AprobacionEntrada) -> AprobacionAceptada:
    estado = await leer(app.state.redis, job_id)
    if estado is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    if estado["status"] != "AWAITING_APPROVAL":
        raise HTTPException(status_code=409, detail="La tarea no está esperando aprobación")
    estado["aprobado"] = entrada.aprobado
    estado["comentario"] = entrada.comentario
    estado["fase"] = "reanudar"
    estado["status"] = "RUNNING"
    await guardar(app.state.redis, job_id, estado)
    await app.state.redis.rpush(QUEUE_NAME, job_id)
    return AprobacionAceptada(job_id=job_id, status="RUNNING")


@app.get("/conversations/{thread_id}", response_model=Conversacion, tags=["conversaciones"])
async def leer_conversacion(thread_id: str) -> Conversacion:
    snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    valores = snapshot.values if snapshot else None
    mensajes = list((valores or {}).get("messages") or [])
    if not mensajes:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return Conversacion(
        thread_id=thread_id,
        mensajes=[
            MensajeConversacion(
                rol=mensaje.type,
                contenido=texto(mensaje),
                nombre=getattr(mensaje, "name", None),
            )
            for mensaje in mensajes
        ],
        investigacion=(valores or {}).get("investigacion") or "",
        analisis=(valores or {}).get("analisis") or "",
        respuesta_final=(valores or {}).get("respuesta_final") or "",
        ajuste=(valores or {}).get("ajuste") or "",
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
