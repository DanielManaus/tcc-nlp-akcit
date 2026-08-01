"""Cadeias RAG e baseline (sem recuperação).

Ambas usam o MESMO LLM (OpenRouter), MESMA temperatura e MESMO template
de prompt. A única variável independente é a presença do mecanismo de
recuperação (TCC §6).
"""
from functools import lru_cache
import math
import re
import unicodedata

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres import PGVector

from src.config import (
    COLLECTION_NAME,
    HISTORY_COLLECTION_NAME,
    JURISPRUDENCE_COLLECTION_NAME,
    OPENROUTER_MODEL,
    SYSTEM_PROMPT,
    connection_string,
)
from src.embeddings import get_embeddings
from src.models import get_llm


STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "de", "do", "dos", "das",
    "e", "em", "no", "na", "nos", "nas", "o", "os", "ou", "para", "por",
    "qual", "quais", "que", "um", "uma", "sobre", "funciona", "fale",
}

QUERY_EXPANSIONS = [
    (
        ("arrepend", "desistencia", "desistir", "compra online", "internet"),
        "desistir contrato prazo 7 dias fora do estabelecimento telefone domicilio "
        "prazo de reflexao valores devolvidos Art. 49",
    ),
    (
        ("direitos basicos",),
        "direitos basicos consumidor protecao vida saude seguranca informacao "
        "publicidade enganosa inversao onus prova Art. 6",
    ),
    (
        ("troca", "defeito", "produto defeito", "vicio produto"),
        "vicio do produto sanar trinta dias substituicao restituicao abatimento Art. 18",
    ),
    (
        ("garantia", "vicio oculto", "reclamacao"),
        "garantia legal vicio aparente facil constatacao vicio oculto prazo decadencia Art. 26 Art. 50",
    ),
    (
        ("publicidade enganosa", "propaganda enganosa"),
        "publicidade enganosa abusiva informacao inteira precisa ostensiva Art. 36 Art. 37",
    ),
    (
        ("cobranca indevida", "cobrado indevido", "divida"),
        "cobranca debitos repeticao do indebido valor dobro constrangimento ameaca Art. 42",
    ),
    (
        ("banco", "bancario", "financeira", "financiamento", "emprestimo"),
        "servico natureza bancaria financeira credito securitaria fornecedor "
        "instituicoes financeiras contratos bancarios CDC Art. 3 Sumula 297 STJ",
    ),
    (
        ("fraude", "golpe", "pix", "transacao indevida", "cartao clonado"),
        "fraude bancaria fortuito interno responsabilidade objetiva Sumula 479 STJ Art. 14",
    ),
    (
        ("cartao nao solicitado", "cartao sem solicitar", "cartao sem pedir"),
        "envio cartao credito sem solicitacao pratica abusiva Sumula 532 STJ Art. 39",
    ),
    (
        ("plano de saude", "convenio medico", "operadora de saude"),
        "servico fornecedor plano de saude CDC Art. 3 autogestao Sumula 608 STJ",
    ),
    (
        ("serasa", "spc", "nome sujo", "negativacao", "inadimplentes"),
        "cadastro inadimplentes divida paga exclusao registro cinco dias uteis Sumula 548 STJ Art. 43",
    ),
    (
        ("imovel", "construtora", "incorporadora", "distrato"),
        "promessa compra venda imovel prestacoes clausula perda total Art. 53 restituicao parcelas Sumula 543 STJ",
    ),
    (
        ("nasceu", "origem", "historia", "historico", "criado", "criação", "criacao", "quando surgiu", "lei 8078", "lei 8.078", "vigencia"),
        "origem historica CDC Constituicao Federal 1988 ADCT Art. 48 Lei 8078 11 setembro 1990 vigencia 11 marco 1991 Art. 118",
    ),
]


def _prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "Modo de geracao: {generation_mode}\n\n"
                "Contexto legal (CDC):\n{cdc_context}\n\n"
                "Contexto historico/institucional do CDC:\n{history_context}\n\n"
                "Contexto jurisprudencial complementar (STJ):\n{stj_context}\n\n"
                "Pergunta: {question}\n\nResposta:"
            ),
        ]
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    """Vector store do CDC indexado no PostgreSQL + pgvector."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection_string(),
        use_jsonb=True,
    )


@lru_cache(maxsize=1)
def get_jurisprudence_vectorstore() -> PGVector:
    """Vector store da jurisprudência/súmulas do STJ."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=JURISPRUDENCE_COLLECTION_NAME,
        connection=connection_string(),
        use_jsonb=True,
    )


@lru_cache(maxsize=1)
def get_history_vectorstore() -> PGVector:
    """Vector store do contexto historico/institucional do CDC."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=HISTORY_COLLECTION_NAME,
        connection=connection_string(),
        use_jsonb=True,
    )


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('artigo', 'N/A')}]\n{d.page_content}" for d in docs
    )


def format_stj_docs(docs) -> str:
    if not docs:
        return "Nenhuma jurisprudência complementar recuperada."
    return "\n\n".join(
        f"[{d.metadata.get('referencia', 'STJ')}]\n{d.page_content}" for d in docs
    )


def format_history_docs(docs) -> str:
    if not docs:
        return "Nenhum contexto historico recuperado."
    return "\n\n".join(
        f"[{d.metadata.get('referencia', 'Historico CDC')}]\n{d.page_content}"
        for d in docs
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", _normalize(text))
        if len(token) > 2 and token not in STOPWORDS
    }


def expand_query(question: str) -> str:
    """Expande termos jurídicos comuns apenas para melhorar a recuperação."""
    normalized = _normalize(question)
    additions = []
    for triggers, expansion in QUERY_EXPANSIONS:
        if any(trigger in normalized for trigger in triggers):
            additions.append(expansion)
    return " ".join([question, *additions])


@lru_cache(maxsize=1)
def get_corpus_documents():
    """Carrega os chunks locais do CDC para busca lexical auxiliar."""
    from src.config import CDC_LOCAL_PATH
    from src.ingestion import _read_text, split_by_article

    return split_by_article(_read_text(CDC_LOCAL_PATH))


@lru_cache(maxsize=1)
def get_jurisprudence_documents():
    """Carrega o corpus curado do STJ para busca lexical auxiliar."""
    from src.config import STJ_LOCAL_PATH
    from src.ingestion import _read_text, split_stj_summaries

    return split_stj_summaries(_read_text(STJ_LOCAL_PATH))


@lru_cache(maxsize=1)
def get_history_documents():
    """Carrega o corpus historico/institucional do CDC para busca lexical."""
    from src.config import HISTORY_LOCAL_PATH
    from src.ingestion import _read_text, split_history_notes

    return split_history_notes(_read_text(HISTORY_LOCAL_PATH))


def _lexical_score(query: str, text: str, article: str) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(f"{article} {text}")
    if not query_tokens or not text_tokens:
        return 0.0

    overlap = query_tokens & text_tokens
    score = len(overlap) / math.sqrt(len(query_tokens))

    normalized_query = _normalize(query)
    normalized_text = _normalize(text)
    for phrase in ("7 dias", "desistir do contrato", "direito de arrependimento"):
        if phrase in normalized_query and phrase in normalized_text:
            score += 2.0

    article_matches = re.findall(
        r"\bart(?:igo)?\.?\s*(\d+(?:-[a-z])?)\b",
        normalized_query,
    )
    if article in {f"Art. {match.upper()}" for match in article_matches}:
        score += 5.0

    return score


def _lexical_search(question: str, k: int = 8):
    expanded_query = expand_query(question)
    scored = []
    for doc in get_corpus_documents():
        article = doc.metadata.get("artigo", "")
        score = _lexical_score(expanded_query, doc.page_content, article)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def _lexical_search_stj(question: str, k: int = 5):
    expanded_query = expand_query(question)
    scored = []
    for doc in get_jurisprudence_documents():
        reference = doc.metadata.get("referencia", "")
        score = _lexical_score(expanded_query, doc.page_content, reference)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def _lexical_search_history(question: str, k: int = 5):
    expanded_query = expand_query(question)
    scored = []
    for doc in get_history_documents():
        reference = doc.metadata.get("referencia", "")
        score = _lexical_score(expanded_query, doc.page_content, reference)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def _unique_documents(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("artigo"), doc.page_content[:120])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def retrieve_documents(question: str, k: int = 5):
    """Recupera trechos do CDC por busca híbrida: vetor + apoio lexical."""
    expanded_query = expand_query(question)
    vector_docs = get_vectorstore().similarity_search(expanded_query, k=max(k * 3, 12))
    lexical_docs = _lexical_search(question, k=max(k * 2, 8))

    combined = _unique_documents([*lexical_docs, *vector_docs])
    ranked = []
    for index, doc in enumerate(combined):
        article = doc.metadata.get("artigo", "")
        lexical = _lexical_score(expanded_query, doc.page_content, article)
        position_bonus = 1 / (index + 1)
        ranked.append((lexical + position_bonus, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def retrieve_jurisprudence(question: str, k: int = 3):
    """Recupera súmulas/jurisprudência STJ como complemento ao CDC."""
    expanded_query = expand_query(question)
    try:
        vector_docs = get_jurisprudence_vectorstore().similarity_search(
            expanded_query,
            k=max(k * 3, 8),
        )
    except Exception:
        vector_docs = []
    lexical_docs = _lexical_search_stj(question, k=max(k * 2, 5))

    combined = _unique_documents([*lexical_docs, *vector_docs])
    ranked = []
    for index, doc in enumerate(combined):
        reference = doc.metadata.get("referencia", "")
        lexical = _lexical_score(expanded_query, doc.page_content, reference)
        position_bonus = 1 / (index + 1)
        ranked.append((lexical + position_bonus, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def retrieve_history(question: str, k: int = 3):
    """Recupera contexto historico/institucional do CDC."""
    expanded_query = expand_query(question)
    try:
        vector_docs = get_history_vectorstore().similarity_search(
            expanded_query,
            k=max(k * 3, 8),
        )
    except Exception:
        vector_docs = []
    lexical_docs = _lexical_search_history(question, k=max(k * 2, 5))

    combined = _unique_documents([*lexical_docs, *vector_docs])
    ranked = []
    for index, doc in enumerate(combined):
        reference = doc.metadata.get("referencia", "")
        lexical = _lexical_score(expanded_query, doc.page_content, reference)
        position_bonus = 1 / (index + 1)
        ranked.append((lexical + position_bonus, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def answer_question(
    question: str,
    use_rag: bool = True,
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Gera resposta e retorna metadados do modelo efetivamente usado."""
    docs = retrieve_documents(question) if use_rag else []
    history_docs = retrieve_history(question) if use_rag else []
    stj_docs = retrieve_jurisprudence(question) if use_rag else []
    cdc_context = format_docs(docs) if use_rag else "SEM_CONTEXTO_BASELINE"
    history_context = (
        format_history_docs(history_docs) if use_rag else "SEM_CONTEXTO_BASELINE"
    )
    stj_context = format_stj_docs(stj_docs) if use_rag else "SEM_CONTEXTO_BASELINE"
    selected_model = model_name or OPENROUTER_MODEL

    prompt_value = _prompt_template().invoke(
        {
            "generation_mode": "RAG" if use_rag else "Baseline",
            "cdc_context": cdc_context,
            "history_context": history_context,
            "stj_context": stj_context,
            "question": question,
        }
    )
    message = get_llm(model_name=selected_model, api_key=api_key).invoke(prompt_value)
    metadata = message.response_metadata or {}
    answer = (message.content or "").strip()
    if not answer:
        raise RuntimeError(
            "O modelo retornou uma resposta vazia. Tente novamente ou selecione "
            "outro modelo gratuito."
        )

    return {
        "answer": answer,
        "configured_model": selected_model,
        "effective_model": metadata.get("model_name") or selected_model,
        "model_provider": metadata.get("model_provider"),
        "token_usage": metadata.get("token_usage"),
        "documents": docs,
        "history_documents": history_docs,
        "jurisprudence_documents": stj_docs,
    }


def build_rag_chain():
    """Condição RAG: recupera trechos relevantes do pgvector."""
    llm = get_llm()

    chain = (
        {
            "generation_mode": RunnablePassthrough() | (lambda _: "RAG"),
            "cdc_context": RunnablePassthrough() | retrieve_documents | format_docs,
            "history_context": RunnablePassthrough()
            | retrieve_history
            | format_history_docs,
            "stj_context": RunnablePassthrough()
            | retrieve_jurisprudence
            | format_stj_docs,
            "question": RunnablePassthrough(),
        }
        | _prompt_template()
        | llm
        | StrOutputParser()
    )
    return chain


def build_baseline_chain():
    """Condição baseline: SEM recuperação (apenas conhecimento do modelo)."""
    llm = get_llm()
    # contexto vazio para manter o mesmo template
    chain = (
        {
            "generation_mode": RunnablePassthrough() | (lambda _: "Baseline"),
            "cdc_context": RunnablePassthrough()
            | (lambda _: "SEM_CONTEXTO_BASELINE"),
            "history_context": RunnablePassthrough()
            | (lambda _: "SEM_CONTEXTO_BASELINE"),
            "stj_context": RunnablePassthrough()
            | (lambda _: "SEM_CONTEXTO_BASELINE"),
            "question": RunnablePassthrough(),
        }
        | _prompt_template()
        | llm
        | StrOutputParser()
    )
    return chain
