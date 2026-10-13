import re
from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.agents.analyst import nodo_analista
from app.agents.research import nodo_investigador
from app.agents.runtime import pregunta_actual, texto
from app.config import MAX_PASOS, MAX_VISITAS, REDIS_URL
from app.hitl import (
    nodo_aprobacion,
    nodo_clasificar,
    nodo_publicar,
    ruta_tras_aprobacion,
    ruta_tras_clasificar,
)
from app.llm import build_chat_model
from app.schemas import DecisionSupervisor

RUBRICA = (
    "Sos el supervisor de un orquestador jerárquico. No investigás ni calculás.\n"
    "Dada la conversación resumida, elegí quién interviene ahora: "
    "investigador, analista o FINALIZAR.\n"
    "Rúbrica para poder FINALIZAR:\n"
    "1. Tiene que haber evidencia del investigador (reseñas y ficha).\n"
    "2. El análisis tiene que incluir el sentimiento neto y el margen porcentual.\n"
    "Si falta la evidencia, elegí investigador.\n"
    "Si hay evidencia pero falta el análisis, elegí analista.\n"
    "Si ambas están, elegí FINALIZAR, salvo que el análisis esté incompleto: "
    "en ese caso pedí un único refinamiento al analista.\n"
    "No mandes al analista si todavía no hay investigación.\n"
    f"Si los pasos llegan a {MAX_PASOS}, elegí FINALIZAR."
)


def sumar_aportes(actual: list | None, nuevo: list | None) -> list:
    return [*(actual or []), *(nuevo or [])]


class Aporte(TypedDict):
    agente: str
    resumen: str


class OrchestratorState(MessagesState):
    next_agent: Literal["investigador", "analista", "FINALIZAR"]
    task_completed: bool
    pasos: int
    visitas_investigador: int
    visitas_analista: int
    investigacion: str
    analisis: str
    respuesta_final: str
    ultimo_motivo: str
    aportes: Annotated[list[Aporte], sumar_aportes]
    es_critica: bool
    motivo_critico: str
    propuesta_ajuste: dict
    aprobado: bool | None
    comentario_humano: str
    ajuste: str


def analisis_suficiente(texto: str) -> bool:
    bajo = texto.lower()
    tiene_numero = re.search(r"-?\d+(?:[.,]\d+)?", texto) is not None
    return "sentimiento" in bajo and "margen" in bajo and tiene_numero


def aplicar_rubrica(
    decision: str | None, state: OrchestratorState
) -> tuple[Literal["investigador", "analista", "FINALIZAR"], str]:
    if decision not in {"investigador", "analista", "FINALIZAR"}:
        decision = "investigador"

    if state.get("pasos", 0) >= MAX_PASOS:
        return "FINALIZAR", "se alcanzó el tope de pasos"

    tiene_inv = bool((state.get("investigacion") or "").strip())
    tiene_an = analisis_suficiente(state.get("analisis") or "")
    visitas_i = state.get("visitas_investigador", 0)
    visitas_a = state.get("visitas_analista", 0)

    if decision == "analista" and not tiene_inv:
        return "investigador", "el analista no corre sin evidencia del investigador"
    if decision == "investigador" and visitas_i >= MAX_VISITAS:
        return ("analista" if not tiene_an else "FINALIZAR"), "el investigador ya intervino dos veces"
    if decision == "analista" and visitas_a >= MAX_VISITAS:
        return "FINALIZAR", "el analista ya refinó su salida"
    if decision == "FINALIZAR" and not tiene_inv:
        return "investigador", "no se cierra sin investigación"
    if decision == "FINALIZAR" and not tiene_an:
        return "analista", "no se cierra sin sentimiento y margen"
    return decision, "decisión del supervisor"


async def nodo_supervisor(state: OrchestratorState) -> dict:
    resumen = (
        f"Pregunta actual: {pregunta_actual(state)}\n"
        f"Pasos usados: {state.get('pasos', 0)} de {MAX_PASOS}.\n"
        f"Visitas investigador: {state.get('visitas_investigador', 0)}. "
        f"Visitas analista: {state.get('visitas_analista', 0)}.\n"
        f"Investigación:\n{state.get('investigacion') or '(vacío)'}\n\n"
        f"Análisis:\n{state.get('analisis') or '(vacío)'}"
    )
    motivo_modelo = ""
    propuesto: str | None = None
    try:
        decision = await (
            build_chat_model()
            .with_structured_output(DecisionSupervisor)
            .ainvoke(
                [
                    SystemMessage(content=RUBRICA),
                    HumanMessage(content=resumen),
                ]
            )
        )
        if isinstance(decision, dict):
            propuesto = decision.get("next_agent")
            motivo_modelo = decision.get("motivo") or ""
        else:
            propuesto = getattr(decision, "next_agent", None)
            motivo_modelo = getattr(decision, "motivo", "") or ""
    except Exception as exc:
        motivo_modelo = f"el modelo no devolvió una ruta válida ({type(exc).__name__})"

    destino, motivo_rubrica = aplicar_rubrica(propuesto, state)
    if destino == propuesto:
        motivo = motivo_modelo or motivo_rubrica
    else:
        motivo = f"propuesto={propuesto or 'sin respuesta'}; rúbrica={motivo_rubrica}. {motivo_modelo}".strip()

    return {
        "next_agent": destino,
        "ultimo_motivo": motivo,
        "messages": [
            AIMessage(
                content=f"Siguiente: {destino}. {motivo}",
                name="supervisor",
            )
        ],
    }


async def nodo_sintesis(state: OrchestratorState) -> dict:
    pregunta = pregunta_actual(state)
    decision = ""
    if state.get("es_critica"):
        if state.get("aprobado"):
            decision = f"El ajuste de catálogo fue publicado: {state.get('ajuste') or ''}."
        else:
            decision = "La persona rechazó publicar el ajuste. No escribas un precio nuevo."
    try:
        respuesta = await build_chat_model().ainvoke(
            [
                SystemMessage(
                    content=(
                        "Redactá la respuesta final en español, en un párrafo. "
                        "Incluí el número de sentimiento_neto y el de margen_porcentual "
                        "tal como figuran en el bloque Análisis. No recalcules nada. "
                        "Si hay una decisión humana sobre el catálogo, mencionala. "
                        "Si la pregunta no está cubierta por esa evidencia, decilo y no inventes datos."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Pregunta: {pregunta}\n\n"
                        f"Investigación:\n{state.get('investigacion') or '(sin evidencia)'}\n\n"
                        f"Análisis:\n{state.get('analisis') or '(sin análisis)'}\n\n"
                        f"{decision}"
                    )
                ),
            ]
        )
        cierre = texto(respuesta)
    except Exception as exc:
        cierre = (
            "No pude redactar el cierre con el modelo. "
            f"Investigación: {state.get('investigacion') or 'sin evidencia'}. "
            f"Análisis: {state.get('analisis') or 'sin análisis'}. "
            f"Error: {type(exc).__name__}."
        )
    return {
        "respuesta_final": cierre,
        "task_completed": True,
        "messages": [AIMessage(content=cierre, name="sintesis")],
        "aportes": [{"agente": "sintesis", "resumen": cierre[:800]}],
    }


def ruta_supervisor(state: OrchestratorState) -> Literal["investigador", "analista", "clasificar"]:
    if state.get("next_agent") == "FINALIZAR":
        return "clasificar"
    return state["next_agent"]


def build_graph(checkpointer):
    grafo = StateGraph(OrchestratorState)
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("clasificar", nodo_clasificar)
    grafo.add_node("aprobacion", nodo_aprobacion)
    grafo.add_node("publicar", nodo_publicar)
    grafo.add_node("sintesis", nodo_sintesis)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        ruta_supervisor,
        {
            "investigador": "investigador",
            "analista": "analista",
            "clasificar": "clasificar",
        },
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_conditional_edges(
        "clasificar",
        ruta_tras_clasificar,
        {"aprobacion": "aprobacion", "sintesis": "sintesis"},
    )
    grafo.add_conditional_edges(
        "aprobacion",
        ruta_tras_aprobacion,
        {"publicar": "publicar", "sintesis": "sintesis"},
    )
    grafo.add_edge("publicar", "sintesis")
    grafo.add_edge("sintesis", END)
    return grafo.compile(checkpointer=checkpointer)


@asynccontextmanager
async def memoria_redis(url: str = REDIS_URL):
    async with AsyncRedisSaver.from_conn_string(url) as saver:
        await saver.asetup()
        yield saver
