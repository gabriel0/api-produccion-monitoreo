from langchain_core.messages import BaseMessage, HumanMessage


def texto(mensaje: BaseMessage) -> str:
    content = mensaje.content
    if isinstance(content, list):
        partes: list[str] = []
        for parte in content:
            if isinstance(parte, str):
                partes.append(parte)
            elif isinstance(parte, dict):
                partes.append(str(parte.get("text", parte)))
            else:
                partes.append(str(parte))
        return " ".join(partes).strip()
    return str(content or "").strip()


async def correr_react(agente, instruccion: str) -> dict:
    return await agente.ainvoke(
        {"messages": [HumanMessage(content=instruccion)]},
        config={"recursion_limit": 8},
    )


def observaciones(mensajes: list[BaseMessage]) -> list[str]:
    return [
        f"{getattr(mensaje, 'name', None) or 'herramienta'}: {texto(mensaje)}"
        for mensaje in mensajes
        if mensaje.type == "tool"
    ]


def cierre_del_agente(mensajes: list[BaseMessage]) -> str:
    for mensaje in reversed(mensajes):
        if mensaje.type == "ai" and texto(mensaje) and not getattr(mensaje, "tool_calls", None):
            return texto(mensaje)
    for mensaje in reversed(mensajes):
        if mensaje.type == "ai" and texto(mensaje):
            return texto(mensaje)
    return ""


def pregunta_actual(state: dict) -> str:
    for mensaje in reversed(state.get("messages") or []):
        if getattr(mensaje, "type", None) == "human":
            return str(mensaje.content)
    return ""
