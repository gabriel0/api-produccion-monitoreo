from langchain_core.language_models.chat_models import BaseChatModel

from app.config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
)


def nombre_modelo_chat() -> str:
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    return OLLAMA_MODEL


def build_chat_model() -> BaseChatModel:
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=gemini requiere GOOGLE_API_KEY en el entorno (.env)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
