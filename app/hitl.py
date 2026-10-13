"""Pausa la publicación en el catálogo hasta que llegue una aprobación externa."""

import asyncio
import json
import re
from datetime import datetime, timezone

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from app.agents.runtime import pregunta_actual
from app.config import AJUSTES_PATH, COSTO_FICHA, PRECIO_FICHA
from app.schemas import PublicarAjusteInput

_MARCAS = (
    "publica",
    "descuento",
    "catalogo",
    "ajustar el precio",
    "ajuste de precio",
    "cambiar el precio",
    "poner en oferta",
)


def _normalizar(texto: str) -> str:
    tabla = str.maketrans("áéíóúüñ", "aeiouun")
    return texto.lower().translate(tabla)


def _cifras(texto: str) -> tuple[float, float]:
    normalizado = _normalizar(texto)
    venta = re.search(r"precio de venta (\d+(?:[.,]\d+)?)", normalizado)
    costo = re.search(r"costo de reposicion (\d+(?:[.,]\d+)?)", normalizado)
    precio = float(venta.group(1).replace(",", ".")) if venta else PRECIO_FICHA
    costo_valor = float(costo.group(1).replace(",", ".")) if costo else COSTO_FICHA
    return precio, costo_valor


def _margen(texto: str) -> float | None:
    hallazgo = re.search(
        r"margen[_\s]?porcentual[\"']?\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
        _normalizar(texto),
    )
    if hallazgo is None:
        return None
    return float(hallazgo.group(1).replace(",", "."))


def armar_propuesta(state: dict) -> dict | None:
    """La tarea es crítica si pide escribir en el catálogo."""
    pregunta = pregunta_actual(state)
    if not any(marca in _normalizar(pregunta) for marca in _MARCAS):
        return None
    evidencia = f"{state.get('investigacion') or ''}\n{state.get('analisis') or ''}"
    precio, costo = _cifras(evidencia)
    margen = _margen(state.get("analisis") or "")
    if margen is None and precio:
        margen = round((precio - costo) / precio * 100, 2)
    return {
        "accion": "publicar_ajuste_precio",
        "motivo": "la consulta pide un cambio de precio en el catálogo",
        "precio_actual": precio,
        "precio_propuesto": round(precio * 0.9, 2),
        "margen_porcentual": margen,
    }


def nodo_clasificar(state: dict) -> dict:
    propuesta = armar_propuesta(state)
    if propuesta is None:
        return {"es_critica": False, "motivo_critico": "", "propuesta_ajuste": {}}
    return {
        "es_critica": True,
        "motivo_critico": propuesta["motivo"],
        "propuesta_ajuste": propuesta,
    }


def ruta_tras_clasificar(state: dict) -> str:
    if state.get("es_critica"):
        return "aprobacion"
    return "sintesis"


def nodo_aprobacion(state: dict) -> dict:
    """Se reanuda desde el principio del nodo cuando llega Command(resume=...)."""
    propuesta = state.get("propuesta_ajuste") or {}
    decision = interrupt(
        {
            "accion": propuesta.get("accion") or "publicar_ajuste_precio",
            "motivo": state.get("motivo_critico") or "",
            "precio_actual": float(propuesta.get("precio_actual") or 0),
            "precio_propuesto": float(propuesta.get("precio_propuesto") or 0),
            "margen_porcentual": propuesta.get("margen_porcentual"),
        }
    )
    aprobado = False
    comentario = ""
    if isinstance(decision, dict):
        aprobado = bool(decision.get("aprobado"))
        comentario = str(decision.get("comentario") or "")
    elif isinstance(decision, bool):
        aprobado = decision
    texto = "Aprobado" if aprobado else "Rechazado"
    if comentario:
        texto = f"{texto}. {comentario}"
    return {
        "aprobado": aprobado,
        "comentario_humano": comentario,
        "messages": [AIMessage(content=f"Aprobación humana: {texto}", name="aprobacion")],
    }


def ruta_tras_aprobacion(state: dict) -> str:
    if state.get("aprobado"):
        return "publicar"
    return "sintesis"


@tool(args_schema=PublicarAjusteInput)
async def publicar_ajuste_precio(precio_actual: float, precio_nuevo: float, motivo: str) -> str:
    """Escribe un precio nuevo en el catálogo. Tiene efecto secundario."""
    if precio_nuevo <= 0 or precio_actual <= 0:
        raise ValueError("el precio publicado tiene que ser mayor a 0")
    registro = {
        "sku": "nova-x",
        "precio_anterior": precio_actual,
        "precio_nuevo": precio_nuevo,
        "motivo": motivo,
        "publicado_en": datetime.now(timezone.utc).isoformat(),
    }
    linea = json.dumps(registro, ensure_ascii=False)

    def _escribir() -> None:
        AJUSTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AJUSTES_PATH.open("a", encoding="utf-8") as archivo:
            archivo.write(linea + "\n")

    await asyncio.to_thread(_escribir)
    return linea


async def nodo_publicar(state: dict) -> dict:
    propuesta = state.get("propuesta_ajuste") or {}
    resultado = await publicar_ajuste_precio.ainvoke(
        {
            "precio_actual": float(propuesta.get("precio_actual") or PRECIO_FICHA),
            "precio_nuevo": float(propuesta.get("precio_propuesto") or 0),
            "motivo": state.get("motivo_critico") or "ajuste aprobado",
        }
    )
    return {
        "ajuste": resultado,
        "aportes": [{"agente": "publicar", "resumen": resultado[:800]}],
        "messages": [AIMessage(content=resultado, name="publicar")],
        "pasos": state.get("pasos", 0) + 1,
    }
