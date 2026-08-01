"""Cliente LLM via OpenRouter (API OpenAI-compatível).

Substitui o Qwen 2.5 local do TCC por um modelo gratuito do OpenRouter,
mantendo a compatibilidade com LangChain (ChatOpenAI).
"""
from functools import lru_cache

import requests
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    LLM_TEMPERATURE,
)

CURATED_FREE_MODELS = [
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "NVIDIA: Nemotron 3 Ultra (free)",
        "context_length": 1000000,
        "note": "Recomendado para RAG: melhor contexto e raciocinio.",
    },
    {
        "id": "inclusionai/ling-3.0-flash:free",
        "name": "Ling-3.0-flash (free)",
        "context_length": 262144,
        "note": "Rapido e eficiente em tokens.",
    },
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (free)",
        "context_length": 262144,
        "note": "Bom modelo geral de instrucao.",
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "NVIDIA: Nemotron 3 Super (free)",
        "context_length": 262144,
        "note": "Alternativa forte para respostas mais complexas.",
    },
    {
        "id": "google/gemma-4-26b-a4b-it:free",
        "name": "Google: Gemma 4 26B A4B (free)",
        "context_length": 262144,
        "note": "Fallback geral.",
    },
]

BLOCKED_MODEL_TERMS = (
    "safety",
    "guard",
    "moderation",
    "audio",
    "image",
    "vision",
    "omni",
)


def list_free_models() -> list[dict]:
    """Lista apenas os modelos gratuitos curados para a POC."""
    headers = {"User-Agent": "RAG-CDC-POC"}
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL.rstrip('/')}/models",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json().get("data", [])
    except Exception:
        return CURATED_FREE_MODELS

    available_by_id = {}
    for row in rows:
        model_id = row.get("id", "")
        name = row.get("name") or model_id
        architecture = row.get("architecture") or {}
        output_modalities = architecture.get("output_modalities") or ["text"]
        label = f"{model_id} {name}".lower()

        if not model_id.endswith(":free"):
            continue
        if "text" not in output_modalities:
            continue
        if any(term in label for term in BLOCKED_MODEL_TERMS):
            continue

        available_by_id[model_id] = {
            "id": model_id,
            "name": name,
            "context_length": row.get("context_length"),
        }

    models = []
    for curated in CURATED_FREE_MODELS:
        if curated["id"] not in available_by_id:
            continue
        model = available_by_id[curated["id"]]
        model["note"] = curated.get("note", "")
        models.append(model)

    if not models:
        return CURATED_FREE_MODELS

    return models


@lru_cache(maxsize=16)
def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Retorna o chat model configurado para o OpenRouter.

    Levanta RuntimeError se a chave de API não estiver definida.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY não definida. Defina no .env ou no ambiente."
        )
    selected_model = model_name or OPENROUTER_MODEL
    extra_kwargs = {}
    if selected_model.startswith("openai/gpt-oss"):
        # Modelos gpt-oss usam tokens de raciocínio interno. Em "low",
        # sobra orçamento para a resposta visível.
        extra_kwargs["reasoning_effort"] = "low"

    return ChatOpenAI(
        model=selected_model,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-OpenRouter-Title": OPENROUTER_APP_NAME,
        },
        max_tokens=LLM_MAX_TOKENS,
        timeout=120,
        **extra_kwargs,
    )
