"""Búsqueda vectorial local. El embedding va por HTTP y el coseno queda aparte."""

import json
import time

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import (
    CORPUS_PATH,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
    VECTORES_PATH,
    VERSION_PROMPT,
)
from app.schemas import Fragmento

_tracer = trace.get_tracer("preentrega7.vector_store")
_cliente: httpx.AsyncClient | None = None
_indice_cache: list[dict] | None = None


def cliente_http() -> httpx.AsyncClient:
    global _cliente
    if _cliente is None:
        _cliente = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _cliente


async def cerrar_http() -> None:
    global _cliente
    if _cliente is not None:
        await _cliente.aclose()
        _cliente = None


def _coseno(izquierda: list[float], derecha: list[float]) -> float:
    producto = sum(a * b for a, b in zip(izquierda, derecha))
    norma_izq = sum(a * a for a in izquierda) ** 0.5
    norma_der = sum(b * b for b in derecha) ** 0.5
    return producto / (norma_izq * norma_der + 1e-9)


async def _embed(texto: str) -> list[float]:
    response = await cliente_http().post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texto},
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


async def indice() -> list[dict]:
    global _indice_cache
    if _indice_cache is not None:
        return _indice_cache

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if VECTORES_PATH.exists():
        guardados = json.loads(VECTORES_PATH.read_text(encoding="utf-8"))
        if guardados.get("modelo") == EMBED_MODEL and len(guardados.get("items", [])) == len(corpus):
            _indice_cache = guardados["items"]
            return _indice_cache

    items = []
    with _tracer.start_as_current_span("vector_db.construir_indice") as span:
        span.set_attribute("openinference.span.kind", "EMBEDDING")
        inicio = time.perf_counter()
        for doc in corpus:
            items.append({**doc, "vector": await _embed(doc["texto"])})
        span.set_attribute("latency.network_ms", round((time.perf_counter() - inicio) * 1000, 2))
        span.set_attribute("embedding.model_name", EMBED_MODEL)
        span.set_attribute("retrieval.document_count", len(items))
    VECTORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORES_PATH.write_text(
        json.dumps({"modelo": EMBED_MODEL, "items": items}, ensure_ascii=False),
        encoding="utf-8",
    )
    _indice_cache = items
    return items


async def buscar(consulta: str, k: int = TOP_K) -> list[Fragmento]:
    with _tracer.start_as_current_span("vector_db.buscar") as span:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("input.value", consulta)
        span.set_attribute("input.mime_type", "text/plain")
        span.set_attribute("prompt_template_version", VERSION_PROMPT)
        try:
            red = time.perf_counter()
            with _tracer.start_as_current_span("ollama.embed") as embed_span:
                embed_span.set_attribute("openinference.span.kind", "EMBEDDING")
                embed_span.set_attribute("embedding.model_name", EMBED_MODEL)
                embed_span.set_attribute("input.value", consulta)
                vector = await _embed(consulta)
                embed_span.set_attribute("embedding.vector_dim", len(vector))
            red_ms = (time.perf_counter() - red) * 1000

            documentos = await indice()
            proceso = time.perf_counter()
            ranking = sorted(
                (
                    {
                        "id": doc["id"],
                        "fuente": doc["fuente"],
                        "texto": doc["texto"],
                        "score": max(-1.0, min(1.0, round(_coseno(vector, doc["vector"]), 4))),
                    }
                    for doc in documentos
                ),
                key=lambda doc: doc["score"],
                reverse=True,
            )
            proc_ms = (time.perf_counter() - proceso) * 1000
            fragmentos = [Fragmento.model_validate(doc) for doc in ranking[:k]]
            span.set_attribute("latency.network_ms", round(red_ms, 2))
            span.set_attribute("latency.processing_ms", round(proc_ms, 2))
            span.set_attribute(
                "output.value",
                json.dumps([item.model_dump() for item in fragmentos], ensure_ascii=False),
            )
            for posicion, doc in enumerate(fragmentos):
                span.set_attribute(f"retrieval.documents.{posicion}.document.id", doc.id)
                span.set_attribute(f"retrieval.documents.{posicion}.document.content", doc.texto)
                span.set_attribute(f"retrieval.documents.{posicion}.document.score", doc.score)
            span.set_status(Status(StatusCode.OK))
            return fragmentos
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
