import json

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.agents.runtime import correr_react, observaciones, pregunta_actual
from app.agents.vector_store import buscar
from app.llm import build_chat_model
from app.schemas import BuscarFuentesInput

PROMPT = (
    "Sos el investigador de un orquestador. Tu único trabajo es recuperar evidencia.\n"
    "Tenés una sola herramienta, buscar_fuentes. Llamala con la consulta antes de responder.\n"
    "En el cierre, citá las fuentes recuperadas y no calcules sentimiento ni margen."
)


@tool(args_schema=BuscarFuentesInput)
async def buscar_fuentes(consulta: str) -> str:
    """Busca reseñas y fichas en la base vectorial local."""
    hallazgos = await buscar(consulta)
    return json.dumps([item.model_dump() for item in hallazgos], ensure_ascii=False)


investigador = create_react_agent(
    build_chat_model(),
    tools=[buscar_fuentes],
    prompt=PROMPT,
    name="investigador",
)


async def nodo_investigador(state) -> dict:
    pregunta = pregunta_actual(state)
    instruccion = (
        f"Consulta del usuario: {pregunta}\n"
        "Recuperá reseñas del producto y la ficha con precio de venta y costo de reposición."
    )
    resultado = await correr_react(investigador, instruccion)
    notas = observaciones(resultado["messages"])
    if not notas:
        notas = [f"buscar_fuentes: {await buscar_fuentes.ainvoke({'consulta': pregunta})}"]
    evidencia = "\n\n".join(notas).strip()
    return {
        "investigacion": evidencia,
        "aportes": [{"agente": "investigador", "resumen": evidencia[:800]}],
        "messages": [AIMessage(content=evidencia[:1200], name="investigador")],
        "pasos": state.get("pasos", 0) + 1,
        "visitas_investigador": state.get("visitas_investigador", 0) + 1,
    }
