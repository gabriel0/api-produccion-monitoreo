"""Trazas OpenInference hacia Phoenix. iniciar() corre antes de importar LangChain."""

import logging
import re

import httpx
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from app.config import (
    COSTO_ENTRADA_POR_MILLON,
    COSTO_SALIDA_POR_MILLON,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    PHOENIX_ENDPOINT,
    PHOENIX_PROJECT,
    PHOENIX_URL,
)

log = logging.getLogger("preentrega7")
_provider = None


def _nombre_modelo() -> str:
    return GEMINI_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_MODEL

_MUTACION = """
mutation ($input: CreateModelMutationInput!) {
  createModel(input: $input) {
    model { name }
  }
}
"""


def iniciar():
    global _provider
    if _provider is not None:
        return _provider
    _provider = register(
        project_name=PHOENIX_PROJECT,
        endpoint=PHOENIX_ENDPOINT,
        batch=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()


async def registrar_tarifa() -> None:
    """Phoenix calcula el costo con esta tarifa y los tokens de cada span."""
    modelo = _nombre_modelo()
    proveedor = "google" if LLM_PROVIDER == "gemini" else "ollama"
    variables = {
        "input": {
            "name": modelo,
            "provider": proveedor,
            "namePattern": re.escape(modelo),
            "costs": [
                {
                    "tokenType": "input",
                    "costPerMillionTokens": COSTO_ENTRADA_POR_MILLON,
                    "kind": "PROMPT",
                },
                {
                    "tokenType": "output",
                    "costPerMillionTokens": COSTO_SALIDA_POR_MILLON,
                    "kind": "COMPLETION",
                },
            ],
        }
    }
    try:
        async with httpx.AsyncClient(timeout=10) as cliente:
            respuesta = await cliente.post(
                f"{PHOENIX_URL}/graphql",
                json={"query": _MUTACION, "variables": variables},
            )
        cuerpo = respuesta.json()
    except Exception as exc:
        log.warning("No pude registrar la tarifa en Phoenix: %s", exc)
        return
    errores = cuerpo.get("errors") or []
    if errores and any("already exists" in (item.get("message") or "") for item in errores):
        log.info("La tarifa de %s ya estaba en Phoenix", modelo)
        return
    if errores or respuesta.status_code >= 400:
        log.warning("Phoenix no aceptó la tarifa: %s", cuerpo)
        return
    log.info("Tarifa de %s (%s) registrada en Phoenix", modelo, proveedor)
