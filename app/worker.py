import asyncio
import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import JOB_PREFIX, JOB_TIMEOUT, QUEUE_NAME, VERSION_PROMPT
from app.observability import flush

log = logging.getLogger("preentrega7")
_tracer = trace.get_tracer("preentrega7.worker")


def clave(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"


async def guardar(redis, job_id: str, estado: dict) -> None:
    await redis.set(clave(job_id), json.dumps(estado, ensure_ascii=False))


async def leer(redis, job_id: str) -> dict | None:
    crudo = await redis.get(clave(job_id))
    if crudo is None:
        return None
    return json.loads(crudo)


def _valores(resultado) -> dict:
    if isinstance(resultado, dict):
        return resultado
    valor = getattr(resultado, "value", None)
    if isinstance(valor, dict):
        return valor
    return {}


def _pausa(resultado, snapshot) -> dict | None:
    candidatos = []
    propios = getattr(resultado, "interrupts", None)
    if propios:
        candidatos.extend(propios)
    if isinstance(resultado, dict):
        candidatos.extend(resultado.get("__interrupt__") or [])
    candidatos.extend(getattr(snapshot, "interrupts", ()) or ())
    for task in getattr(snapshot, "tasks", ()) or ():
        candidatos.extend(getattr(task, "interrupts", ()) or ())
    for item in candidatos:
        valor = getattr(item, "value", item)
        if isinstance(valor, dict):
            return valor
    return None


async def _correr(graph, estado: dict, reanudar: bool) -> dict:
    config = {
        "configurable": {"thread_id": estado["thread_id"]},
        "recursion_limit": 40,
        "metadata": {
            "job_id": estado["job_id"],
            "prompt_template_version": VERSION_PROMPT,
        },
        "tags": ["pre-entrega7"],
    }
    if reanudar:
        entrada = Command(
            resume={
                "aprobado": bool(estado.get("aprobado")),
                "comentario": estado.get("comentario") or "",
            }
        )
    else:
        entrada = {"messages": [HumanMessage(content=estado["consulta"])]}

    with _tracer.start_as_current_span("orquestador.consulta") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("session.id", estado["thread_id"])
        span.set_attribute("input.value", estado["consulta"])
        span.set_attribute("input.mime_type", "text/plain")
        span.set_attribute("prompt_template_version", VERSION_PROMPT)
        span.set_attribute("job.id", estado["job_id"])
        try:
            try:
                resultado = await asyncio.wait_for(
                    graph.ainvoke(entrada, config=config, durability="sync"),
                    timeout=JOB_TIMEOUT,
                )
            except GraphInterrupt:
                resultado = {}
            snapshot = await graph.aget_state(config)
            pausa = _pausa(resultado, snapshot)
            if pausa is not None:
                span.set_attribute("hitl.estado", "AWAITING_APPROVAL")
                span.set_attribute("output.value", json.dumps(pausa, ensure_ascii=False))
                span.set_status(Status(StatusCode.OK))
                return {"pausado": True, "propuesta": pausa}
            if getattr(snapshot, "next", ()):
                raise RuntimeError(
                    "el grafo se detuvo en "
                    + ", ".join(snapshot.next)
                    + " sin una aprobación pendiente"
                )
            valores = _valores(resultado) or (snapshot.values or {})
            texto = (valores.get("respuesta_final") or "").strip()
            span.set_attribute("output.value", texto)
            span.set_attribute("hitl.estado", "DONE")
            span.set_status(Status(StatusCode.OK))
            return {
                "pausado": False,
                "respuesta": texto,
                "pasos": valores.get("pasos"),
                "es_critica": bool(valores.get("es_critica")),
                "aprobado": valores.get("aprobado"),
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            flush()


async def ejecutar(graph, redis, job_id: str) -> None:
    estado = await leer(redis, job_id)
    if estado is None:
        return
    reanudar = estado.get("fase") == "reanudar"
    if estado["status"] not in {"PENDING", "RUNNING"}:
        return
    estado["status"] = "RUNNING"
    estado["error"] = None
    await guardar(redis, job_id, estado)
    try:
        cierre = await _correr(graph, estado, reanudar)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("Falló el job %s", job_id)
        estado["status"] = "FAILED"
        estado["error"] = f"{type(exc).__name__}: {exc}"
        await guardar(redis, job_id, estado)
        return
    if cierre["pausado"]:
        estado["status"] = "AWAITING_APPROVAL"
        estado["fase"] = "pausada"
        estado["es_critica"] = True
        estado["propuesta"] = cierre["propuesta"]
        estado["error"] = None
    else:
        estado["status"] = "DONE"
        estado["fase"] = "cerrada"
        estado["respuesta"] = cierre["respuesta"]
        estado["pasos"] = cierre["pasos"]
        estado["es_critica"] = cierre["es_critica"]
        estado["aprobado"] = cierre["aprobado"]
        estado["error"] = None
    await guardar(redis, job_id, estado)


async def consumir(app) -> None:
    log.info("Worker de tareas iniciado")
    redis = app.state.redis_worker
    while True:
        try:
            item = await redis.blpop(QUEUE_NAME, timeout=5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("No pude leer la cola: %s", exc)
            await asyncio.sleep(1)
            continue
        if item is None:
            continue
        _, job_id = item
        try:
            await ejecutar(app.state.graph, redis, job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("El worker no pudo cerrar el job %s", job_id)
