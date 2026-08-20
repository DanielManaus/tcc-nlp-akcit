"""Configuração central do projeto RAG-CDC (POC).

Lê variáveis de ambiente (Docker / .env / run.sh). O LLM é acessado via
OpenRouter (API compatível com OpenAI), usando um modelo gratuito.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- OpenRouter / LLM ---
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = [
    _get("OPENROUTER_API_KEY_1", "") or OPENROUTER_API_KEY,
    _get("OPENROUTER_API_KEY_2", ""),
    _get("OPENROUTER_API_KEY_3", ""),
]
OPENROUTER_MODEL = _get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_BASE_URL = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = _get("OPENROUTER_SITE_URL", "https://localhost:8501")
OPENROUTER_APP_NAME = _get("OPENROUTER_APP_NAME", "RAG-CDC-POC")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "1800"))

# --- PostgreSQL / pgvector ---
POSTGRES_HOST = _get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(_get("POSTGRES_PORT", "5432"))
POSTGRES_DB = _get("POSTGRES_DB", "ragcdc")
POSTGRES_USER = _get("POSTGRES_USER", "ragcdc")
POSTGRES_PASSWORD = _get("POSTGRES_PASSWORD", "ragcdc")

# --- Embeddings ---
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = int(_get("EMBEDDING_DIM", "384"))

# --- Corpus CDC ---
CDC_URL = _get(
    "CDC_URL",
    "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
)
CDC_LOCAL_PATH = _get("CDC_LOCAL_PATH", _get("CDC_LOCAL_PDF", "data/cdc.md"))
STJ_LOCAL_PATH = _get("STJ_LOCAL_PATH", "corpus/stj_sumulas_consumidor.md")
HISTORY_LOCAL_PATH = _get("HISTORY_LOCAL_PATH", "corpus/historico_cdc.md")

# Coleção de vetores no pgvector
COLLECTION_NAME = "cdc_chunks"
JURISPRUDENCE_COLLECTION_NAME = "stj_jurisprudencia"
HISTORY_COLLECTION_NAME = "cdc_historico"

# Prompt template (idêntico para RAG e baseline, conforme TCC §6).
SYSTEM_PROMPT = (
    "Voce e um assistente juridico especializado EXCLUSIVAMENTE no Codigo de "
    "Defesa do Consumidor (CDC, Lei n. 8.078/1990) do Brasil.\n"
    "Regras obrigatorias:\n"
    "0. Observe o campo 'Modo de geracao'. No modo RAG, use somente os "
    "trechos recuperados. No modo Baseline, responda com conhecimento "
    "parametrico do modelo, sem recuperacao.\n"
    "1. Quando o contexto trouxer trechos do CDC, contexto historico ou jurisprudencia do STJ, "
    "responda SOMENTE com base nesses trechos recuperados.\n"
    "2. Quando o contexto for 'SEM_CONTEXTO_BASELINE', responda com o "
    "conhecimento parametrico do modelo; essa condicao e usada apenas para "
    "comparacao experimental sem RAG.\n"
    "3. Para perguntas sobre direitos, deveres ou regras do CDC, comece com: "
    "'Fundamento: Art. X.' ou 'Fundamento: Arts. X e Y.'.\n"
    "4. CITE obrigatoriamente o(s) numero(s) do(s) artigo(s) do CDC usado(s) "
    "para fundamentar a resposta, usando no maximo os 2 artigos mais aderentes "
    "ao caso. Nao cite artigo so porque ele apareceu em contexto acessorio.\n"
    "4.1. Quando houver contexto suficiente, nao responda em apenas uma frase. "
    "Explique em linguagem simples e didatica, com 3 a 5 paragrafos ou topicos "
    "curtos, cobrindo: regra legal, explicacao pratica, consequencia para o "
    "consumidor e providencias recomendadas. Essa regra nao se aplica a recusa "
    "obrigatoria por falta de contexto.\n"
    "5. Se houver jurisprudencia/sumula do STJ no contexto, use-a apenas como "
    "complemento e identifique em secao curta: 'Jurisprudencia relacionada "
    "(STJ): Sumula X'. Nunca trate sumula como artigo de lei.\n"
    "5.1. Se a pergunta for historica ou institucional, comece com 'Fonte: ...', "
    "use o contexto historico recuperado e identifique a fonte como "
    "Constituicao Federal, ADCT ou Lei 8.078/1990.\n"
    "6. No modo RAG, se a informacao NAO consta no contexto recuperado, ou se "
    "a pergunta nao for sobre o CDC, responda "
    "EXATAMENTE: 'A informacao nao consta no Codigo de Defesa do Consumidor.'\n"
    "7. Nao invente artigos, sumulas, precedentes ou fontes. Seja claro, "
    "objetivo e util para uma pessoa leiga.\n"
)


def connection_string() -> str:
    """String SQLAlchemy/psycopg usada pelo LangChain PGVector."""
    return (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def psycopg_connection_string() -> str:
    """String nativa do psycopg para consultas diretas no PostgreSQL."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def has_llm_credentials() -> bool:
    return any(OPENROUTER_API_KEYS)


def openrouter_api_key_options() -> list[dict]:
    """Retorna chaves configuradas sem expor os valores na interface."""
    return [
        {
            "label": f"Chave API {index}",
            "api_key": api_key,
        }
        for index, api_key in enumerate(OPENROUTER_API_KEYS, start=1)
        if api_key
    ]
