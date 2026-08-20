"""Cliente LLM via OpenRouter (API OpenAI-compatível).

Substitui o Qwen 2.5 local do TCC por um modelo gratuito do OpenRouter,
mantendo a compatibilidade com LangChain (ChatOpenAI).
"""

import requests
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MAX_TOKENS,
    OPENROUTER_API_KEYS,
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
        "display_name": "Nemotron 3 Ultra",
        "parameters": "550B total / 55B ativos",
        "tier": "Grátis",
        "context_length": 1000000,
        "note": "Recomendado para RAG: melhor contexto e raciocinio.",
    },
    {
        "id": "inclusionai/ling-3.0-flash:free",
        "name": "Ling-3.0-flash (free)",
        "display_name": "Ling 3.0 Flash",
        "parameters": "não divulgado",
        "tier": "Grátis",
        "context_length": 262144,
        "note": "Rapido e eficiente em tokens.",
    },
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (free)",
        "display_name": "Gemma 4 31B",
        "parameters": "31B",
        "tier": "Grátis",
        "context_length": 262144,
        "note": "Bom modelo geral de instrucao.",
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "NVIDIA: Nemotron 3 Super (free)",
        "display_name": "Nemotron 3 Super",
        "parameters": "120B total / 12B ativos",
        "tier": "Grátis",
        "context_length": 262144,
        "note": "Alternativa forte para respostas mais complexas.",
    },
    {
        "id": "google/gemma-4-26b-a4b-it:free",
        "name": "Google: Gemma 4 26B A4B (free)",
        "display_name": "Gemma 4 26B",
        "parameters": "25B total / 3.8B ativos",
        "tier": "Grátis",
        "context_length": 262144,
        "note": "Fallback geral.",
    },
]

CURATED_PAID_MODELS = [
    {
        "id": "openai/gpt-4o-mini",
        "name": "OpenAI: GPT-4o-mini",
        "display_name": "GPT-4o mini",
        "parameters": "não divulgado",
        "tier": "Pago",
        "context_length": 128000,
        "note": (
            "Pago barato: US$ 0,15/1M entrada e US$ 0,60/1M saída "
            "no OpenRouter. Usado para validar a qualidade vs. free."
        ),
    },
    {
        "id": "google/gemini-2.5-flash-lite",
        "name": "Google: Gemini 2.5 Flash Lite",
        "display_name": "Gemini 2.5 Flash Lite",
        "parameters": "não divulgado",
        "tier": "Pago",
        "context_length": 1048576,
        "note": (
            "Pago barato e popular: US$ 0,10/1M entrada e US$ 0,40/1M saída "
            "no OpenRouter. Bom fallback para demo."
        ),
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


def list_free_models(include_paid: bool = True) -> list[dict]:
    """Lista os modelos gratuitos curados para a POC, mais modelos pagos de
    baixo custo quando ``include_paid`` for True (usados para validação)."""
    headers = {"User-Agent": "RAG-CDC-POC"}
    api_key = next((key for key in OPENROUTER_API_KEYS if key), "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL.rstrip('/')}/models",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json().get("data", [])
    except Exception:
        return _curated_fallback(include_paid=include_paid)

    available_by_id = {}
    for row in rows:
        model_id = row.get("id", "")
        name = row.get("name") or model_id
        architecture = row.get("architecture") or {}
        output_modalities = architecture.get("output_modalities") or ["text"]
        label = f"{model_id} {name}".lower()

        is_free = model_id.endswith(":free")
        if not is_free and not include_paid:
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

    curated_sources = [CURATED_FREE_MODELS]
    if include_paid:
        curated_sources.append(CURATED_PAID_MODELS)

    models = []
    for curated_list in curated_sources:
        for curated in curated_list:
            if curated["id"] not in available_by_id:
                continue
            model = available_by_id[curated["id"]]
            model.update(
                {
                    "display_name": curated.get("display_name", model["name"]),
                    "parameters": curated.get("parameters", "não divulgado"),
                    "tier": curated.get("tier", "Grátis"),
                    "note": curated.get("note", ""),
                }
            )
            models.append(model)

    if not models:
        return _curated_fallback(include_paid=include_paid)

    return models


def _curated_fallback(include_paid: bool = True) -> list[dict]:
    """Retorna a lista curada (free + paid) quando a API do OpenRouter falha."""
    models = list(CURATED_FREE_MODELS)
    if include_paid:
        models = models + list(CURATED_PAID_MODELS)
    return models


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Retorna o chat model configurado para o OpenRouter.

    Levanta RuntimeError se a chave de API não estiver definida.
    """
    selected_api_key = api_key or next((key for key in OPENROUTER_API_KEYS if key), "")
    if not selected_api_key:
        raise RuntimeError(
            "Nenhuma chave OpenRouter definida. Configure OPENROUTER_API_KEY_1, "
            "OPENROUTER_API_KEY_2 ou OPENROUTER_API_KEY_3 no .env."
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
        api_key=selected_api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-OpenRouter-Title": OPENROUTER_APP_NAME,
        },
        max_tokens=LLM_MAX_TOKENS,
        timeout=120,
        **extra_kwargs,
    )
