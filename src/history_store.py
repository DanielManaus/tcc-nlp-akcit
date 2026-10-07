"""Persistência do histórico da POC no PostgreSQL/pgvector.

O banco já é usado pelo RAG para embeddings do CDC. Este módulo cria uma
tabela separada para registrar perguntas, respostas, avaliação automática e
avaliação humana, mantendo o histórico disponível mesmo após reiniciar a app.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.config import EMBEDDING_DIM, psycopg_connection_string
from src.embeddings import get_embeddings


TABLE_NAME = "rag_response_history"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _embed_question(question: str) -> str | None:
    try:
        embedding = get_embeddings().embed_query(question)
        return _vector_literal(embedding)
    except Exception:
        return None


def _decimal_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _normalize_row(row: dict) -> dict:
    return {key: _decimal_to_float(value) for key, value in row.items()}


def ensure_history_table() -> None:
    """Cria a tabela de histórico se ela ainda não existir."""
    with psycopg.connect(psycopg_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    question TEXT NOT NULL,
                    question_embedding vector({EMBEDDING_DIM}),
                    mode TEXT NOT NULL,
                    compare_mode BOOLEAN NOT NULL DEFAULT false,
                    response_model TEXT,
                    response_model_label TEXT,
                    api_key_label TEXT,
                    rag_answer TEXT,
                    baseline_answer TEXT,
                    single_answer TEXT,
                    ai_quality JSONB,
                    ai_rag_score NUMERIC(3,1),
                    ai_baseline_score NUMERIC(3,1),
                    ai_winner TEXT,
                    rag_sources JSONB,
                    source_counts JSONB,
                    human_score NUMERIC(3,1)
                        CHECK (human_score IS NULL OR human_score BETWEEN 0 AND 5),
                    human_comment TEXT,
                    human_reviewer TEXT,
                    human_updated_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_created_at
                ON {TABLE_NAME} (created_at DESC);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_human_score
                ON {TABLE_NAME} (human_score);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_question_embedding
                ON {TABLE_NAME}
                USING ivfflat (question_embedding vector_cosine_ops)
                WITH (lists = 20);
                """
            )
        conn.commit()


def serialize_sources(result: dict) -> dict:
    """Extrai uma versão compacta das fontes recuperadas."""

    def pack_docs(docs: list, label_key: str) -> list[dict]:
        rows = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            rows.append(
                {
                    "referencia": (
                        metadata.get(label_key)
                        or metadata.get("referencia")
                        or metadata.get("tema")
                        or "Fonte"
                    ),
                    "trecho": getattr(doc, "page_content", "")[:900],
                    "metadata": metadata,
                }
            )
        return rows

    return {
        "cdc": pack_docs(result.get("documents", []), "artigo"),
        "historico": pack_docs(result.get("history_documents", []), "referencia"),
        "stj": pack_docs(result.get("jurisprudence_documents", []), "referencia"),
    }


def source_counts(result: dict) -> dict:
    return {
        "cdc": len(result.get("documents", [])),
        "historico": len(result.get("history_documents", [])),
        "stj": len(result.get("jurisprudence_documents", [])),
    }


def save_interaction(
    *,
    question: str,
    mode: str,
    compare_mode: bool,
    response_model: str | None,
    response_model_label: str | None,
    api_key_label: str | None,
    rag_result: dict | None = None,
    baseline_result: dict | None = None,
    single_result: dict | None = None,
    ai_quality: dict | None = None,
) -> int:
    """Salva uma interação e retorna o ID criado."""
    ensure_history_table()

    rag_answer = (rag_result or {}).get("answer")
    baseline_answer = (baseline_result or {}).get("answer")
    single_answer = (single_result or {}).get("answer")
    reference_result = rag_result or single_result or {}
    sources = serialize_sources(reference_result)
    counts = source_counts(reference_result)
    ai_rag_score = (ai_quality or {}).get("rag", {}).get("score")
    ai_baseline_score = (ai_quality or {}).get("baseline", {}).get("score")
    ai_winner = (ai_quality or {}).get("winner")
    embedding = _embed_question(question)

    columns = [
        "question",
        "mode",
        "compare_mode",
        "response_model",
        "response_model_label",
        "api_key_label",
        "rag_answer",
        "baseline_answer",
        "single_answer",
        "ai_quality",
        "ai_rag_score",
        "ai_baseline_score",
        "ai_winner",
        "rag_sources",
        "source_counts",
    ]
    values: list[Any] = [
        question,
        mode,
        compare_mode,
        response_model,
        response_model_label,
        api_key_label,
        rag_answer,
        baseline_answer,
        single_answer,
        Jsonb(ai_quality) if ai_quality else None,
        ai_rag_score,
        ai_baseline_score,
        ai_winner,
        Jsonb(sources),
        Jsonb(counts),
    ]
    placeholders = ["%s"] * len(values)
    if embedding:
        columns.append("question_embedding")
        placeholders.append("%s::vector")
        values.append(embedding)

    with psycopg.connect(psycopg_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME} ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                RETURNING id;
                """,
                values,
            )
            row_id = cur.fetchone()[0]
        conn.commit()

    return int(row_id)


def update_human_review(
    interaction_id: int,
    *,
    human_score: float,
    human_comment: str = "",
    human_reviewer: str = "",
) -> None:
    ensure_history_table()
    with psycopg.connect(psycopg_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {TABLE_NAME}
                SET human_score = %s,
                    human_comment = %s,
                    human_reviewer = %s,
                    human_updated_at = now(),
                    updated_at = now()
                WHERE id = %s;
                """,
                (human_score, human_comment.strip(), human_reviewer.strip(), interaction_id),
            )
        conn.commit()


def list_interactions(
    *,
    limit: int = 30,
    search: str = "",
    only_pending_human_review: bool = False,
) -> list[dict]:
    ensure_history_table()
    params: list[Any] = []
    where = []
    order_by = "created_at DESC"

    if only_pending_human_review:
        where.append("human_score IS NULL")

    if search.strip():
        embedding = _embed_question(search.strip())
        if embedding:
            order_by = "question_embedding <=> %s::vector ASC NULLS LAST, created_at DESC"
            params.append(embedding)
        else:
            where.append(
                "(question ILIKE %s OR rag_answer ILIKE %s OR baseline_answer ILIKE %s OR single_answer ILIKE %s)"
            )
            term = f"%{search.strip()}%"
            params.extend([term, term, term, term])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)

    with psycopg.connect(psycopg_connection_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    created_at,
                    question,
                    mode,
                    compare_mode,
                    response_model,
                    response_model_label,
                    api_key_label,
                    rag_answer,
                    baseline_answer,
                    single_answer,
                    ai_quality,
                    ai_rag_score,
                    ai_baseline_score,
                    ai_winner,
                    rag_sources,
                    source_counts,
                    human_score,
                    human_comment,
                    human_reviewer,
                    human_updated_at
                FROM {TABLE_NAME}
                {where_sql}
                ORDER BY {order_by}
                LIMIT %s;
                """,
                params,
            )
            return [_normalize_row(dict(row)) for row in cur.fetchall()]


def history_stats() -> dict:
    ensure_history_table()
    with psycopg.connect(psycopg_connection_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE human_score IS NULL) AS pending_human,
                    ROUND(AVG(human_score), 1) AS avg_human,
                    ROUND(AVG(ai_rag_score), 1) AS avg_ai_rag,
                    ROUND(AVG(ai_baseline_score), 1) AS avg_ai_baseline
                FROM {TABLE_NAME};
                """
            )
            row = dict(cur.fetchone())
    return _normalize_row(row)
