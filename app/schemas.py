from typing import Literal

from pydantic import BaseModel, Field


Estado = Literal["PENDING", "RUNNING", "AWAITING_APPROVAL", "DONE", "FAILED"]


class Fragmento(BaseModel):
    id: str
    fuente: str
    texto: str
    score: float = Field(ge=-1, le=1)


class Sentimiento(BaseModel):
    positivos: int = Field(ge=0)
    negativos: int = Field(ge=0)
    sentimiento_neto: float = Field(ge=-1, le=1)


class Margen(BaseModel):
    precio_venta: float = Field(gt=0)
    costo: float = Field(ge=0)
    margen_porcentual: float


class ErrorHerramienta(BaseModel):
    error: str


class BuscarFuentesInput(BaseModel):
    consulta: str = Field(min_length=1, max_length=500)


class SentimientoInput(BaseModel):
    texto: str = Field(min_length=1)


class MargenInput(BaseModel):
    precio_venta: float = Field(gt=0)
    costo: float = Field(ge=0)


class PublicarAjusteInput(BaseModel):
    precio_actual: float = Field(gt=0)
    precio_nuevo: float = Field(gt=0)
    motivo: str = Field(min_length=1, max_length=500)


class DecisionSupervisor(BaseModel):
    next_agent: Literal["investigador", "analista", "FINALIZAR"]
    motivo: str = Field(min_length=1)


class TareaEntrada(BaseModel):
    consulta: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.:-]{1,64}$",
    )


class TareaAceptada(BaseModel):
    job_id: str
    thread_id: str
    status: Literal["PENDING"]


class Propuesta(BaseModel):
    accion: str
    motivo: str
    precio_actual: float
    precio_propuesto: float
    margen_porcentual: float | None = None


class EstadoJob(BaseModel):
    job_id: str
    thread_id: str
    status: Estado
    consulta: str
    respuesta: str | None = None
    pasos: int | None = None
    error: str | None = None
    es_critica: bool = False
    propuesta: Propuesta | None = None
    aprobado: bool | None = None


class AprobacionEntrada(BaseModel):
    aprobado: bool
    comentario: str = Field(default="", max_length=500)


class AprobacionAceptada(BaseModel):
    job_id: str
    status: Literal["RUNNING"]


class MensajeConversacion(BaseModel):
    rol: str
    contenido: str
    nombre: str | None = None


class Conversacion(BaseModel):
    thread_id: str
    mensajes: list[MensajeConversacion]
    investigacion: str
    analisis: str
    respuesta_final: str
    ajuste: str


class Salud(BaseModel):
    api: Literal["ok"]
    redis: Literal["ok", "error"]
