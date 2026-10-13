import re

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.agents.runtime import cierre_del_agente, correr_react, observaciones, pregunta_actual
from app.llm import build_chat_model
from app.schemas import ErrorHerramienta, Margen, MargenInput, Sentimiento, SentimientoInput

PROMPT = (
    "Sos el analista. No inventes cifras: salen de tus herramientas.\n"
    "Llamá analizar_sentimiento con el texto de las reseñas.\n"
    "Llamá calcular_margen con el precio de venta y el costo de reposición de la ficha.\n"
    "Cerrá nombrando el sentimiento neto y el margen porcentual."
)

_POSITIVAS = {"excelente", "conforme", "comodo", "comodos", "bien", "recomiendo", "buena", "bueno", "rapido"}
_NEGATIVAS = {"furioso", "demora", "tarde", "malo", "defectuoso", "queja", "pesimo"}


def _normalizar(texto: str) -> str:
    tabla = str.maketrans("áéíóúñ", "aeioun")
    return texto.lower().translate(tabla)


@tool(args_schema=SentimientoInput)
async def analizar_sentimiento(texto: str) -> str:
    """Cuenta términos positivos y negativos y devuelve el sentimiento neto entre -1 y 1."""
    tokens = re.findall(r"[a-zñ]+", _normalizar(texto))
    positivos = sum(token in _POSITIVAS for token in tokens)
    negativos = sum(token in _NEGATIVAS for token in tokens)
    total = positivos + negativos
    neto = 0.0 if total == 0 else round((positivos - negativos) / total, 2)
    dato = Sentimiento(positivos=positivos, negativos=negativos, sentimiento_neto=neto)
    return dato.model_dump_json()


@tool(args_schema=MargenInput)
async def calcular_margen(precio_venta: float, costo: float) -> str:
    """Margen porcentual: (precio_venta - costo) / precio_venta * 100."""
    if precio_venta <= 0:
        return ErrorHerramienta(error="el precio de venta debe ser mayor a 0").model_dump_json()
    margen = round((precio_venta - costo) / precio_venta * 100, 2)
    dato = Margen(precio_venta=precio_venta, costo=costo, margen_porcentual=margen)
    return dato.model_dump_json()


analista = create_react_agent(
    build_chat_model(),
    tools=[analizar_sentimiento, calcular_margen],
    prompt=PROMPT,
    name="analista",
)


async def _margen_de_respaldo(evidencia: str) -> str:
    normalizado = _normalizar(evidencia)
    venta = re.search(r"precio de venta (\d+)", normalizado)
    costo = re.search(r"costo de reposicion (\d+)", normalizado)
    if not venta or not costo:
        return ErrorHerramienta(error="la evidencia no trae precio y costo").model_dump_json()
    return await calcular_margen.ainvoke(
        {"precio_venta": float(venta.group(1)), "costo": float(costo.group(1))}
    )


async def nodo_analista(state) -> dict:
    pregunta = pregunta_actual(state)
    evidencia = state.get("investigacion") or ""
    refuerzo = ""
    if state.get("visitas_analista", 0) >= 1:
        refuerzo = (
            "\nLa respuesta anterior no alcanzó la rúbrica. "
            "Volvé a llamar las dos herramientas y citá sentimiento_neto y margen_porcentual."
        )
    instruccion = (
        f"Consulta del usuario: {pregunta}\n"
        "Evidencia del investigador, usala tal cual:\n"
        f"{evidencia}"
        f"{refuerzo}"
    )
    resultado = await correr_react(analista, instruccion)
    notas = observaciones(resultado["messages"])
    if not any(nota.startswith("analizar_sentimiento") for nota in notas):
        notas.append(
            f"analizar_sentimiento: {await analizar_sentimiento.ainvoke({'texto': evidencia or pregunta})}"
        )
    if not any(nota.startswith("calcular_margen") for nota in notas):
        notas.append(f"calcular_margen: {await _margen_de_respaldo(evidencia)}")
    cierre = cierre_del_agente(resultado["messages"])
    informe = "\n\n".join([*notas, cierre]).strip()
    return {
        "analisis": informe,
        "aportes": [{"agente": "analista", "resumen": informe[:800]}],
        "messages": [AIMessage(content=cierre or informe[:500], name="analista")],
        "pasos": state.get("pasos", 0) + 1,
        "visitas_analista": state.get("visitas_analista", 0) + 1,
    }
