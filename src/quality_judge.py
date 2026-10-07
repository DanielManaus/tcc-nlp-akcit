"""Avaliador LLM-as-a-Judge para comparar RAG vs baseline.

Usado na demonstração do TCC para dar uma nota de 0 a 5 para cada resposta,
com base na pergunta, nos trechos recuperados e nos critérios de qualidade.
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.models import get_llm


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def _clip(text: str, limit: int = 1400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[...]"


def _doc_label(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        metadata.get("artigo")
        or metadata.get("referencia")
        or metadata.get("tema")
        or "Fonte recuperada"
    )


def _format_docs_for_judge(docs: list, title: str, limit: int) -> str:
    if not docs:
        return f"{title}: nenhum trecho recuperado."

    rows = [f"{title}:"]
    for index, doc in enumerate(docs[:limit], start=1):
        rows.append(
            f"{index}. [{_doc_label(doc)}]\n"
            f"{_clip(getattr(doc, 'page_content', ''), 900)}"
        )
    return "\n\n".join(rows)


def _format_context(rag_result: dict) -> str:
    return "\n\n".join(
        [
            _format_docs_for_judge(
                rag_result.get("documents", []),
                "Trechos legais recuperados do CDC",
                limit=3,
            ),
            _format_docs_for_judge(
                rag_result.get("history_documents", []),
                "Contexto historico/institucional recuperado",
                limit=2,
            ),
            _format_docs_for_judge(
                rag_result.get("jurisprudence_documents", []),
                "Jurisprudencia complementar recuperada",
                limit=2,
            ),
        ]
    )


def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, round(score, 1)))


def _normalize_payload(payload: dict, judge_model: str) -> dict:
    rag = payload.get("rag") or {}
    baseline = payload.get("baseline") or {}
    return {
        "rag": {
            "score": _normalize_score(rag.get("score")),
            "justification": str(rag.get("justification", "")).strip(),
            "strengths": list(rag.get("strengths") or [])[:3],
            "risks": list(rag.get("risks") or [])[:3],
        },
        "baseline": {
            "score": _normalize_score(baseline.get("score")),
            "justification": str(baseline.get("justification", "")).strip(),
            "strengths": list(baseline.get("strengths") or [])[:3],
            "risks": list(baseline.get("risks") or [])[:3],
        },
        "winner": str(payload.get("winner", "")).strip() or "Indefinido",
        "summary": str(payload.get("summary", "")).strip(),
        "judge_model": judge_model,
    }


def evaluate_answers(
    question: str,
    rag_result: dict,
    baseline_result: dict,
    judge_model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Compara duas respostas e retorna avaliação estruturada de 0 a 5."""

    selected_judge_model = judge_model or DEFAULT_JUDGE_MODEL
    context = _format_context(rag_result)
    system = (
        "Voce e um avaliador academico de respostas juridicas em uma POC de NLP/RAG. "
        "Sua funcao e comparar uma resposta gerada com RAG e uma resposta baseline "
        "sem recuperacao. Avalie apenas qualidade, fidelidade ao contexto fornecido, "
        "correcao das citacoes, completude, clareza e risco de alucinacao. "
        "Nao de aconselhamento juridico. Nao invente fontes externas. "
        "Retorne somente JSON valido."
    )
    human = (
        "Pergunta do usuario:\n"
        f"{question}\n\n"
        "Contexto recuperado pelo RAG, usado como base de referencia:\n"
        f"{context}\n\n"
        "Resposta RAG:\n"
        f"{_clip(rag_result.get('answer', ''), 1600)}\n\n"
        "Resposta baseline:\n"
        f"{_clip(baseline_result.get('answer', ''), 1600)}\n\n"
        "Dê uma nota de 0 a 5 para cada resposta:\n"
        "- 5 = correta, fundamentada, clara, completa e sem alucinacao relevante;\n"
        "- 3 = parcialmente correta, mas incompleta ou com fundamento fraco;\n"
        "- 1 = fraca, vaga ou com erro importante;\n"
        "- 0 = fora do escopo, vazia ou majoritariamente incorreta.\n\n"
        "Formato obrigatorio:\n"
        "{\n"
        '  "rag": {"score": 0, "justification": "...", "strengths": ["..."], "risks": ["..."]},\n'
        '  "baseline": {"score": 0, "justification": "...", "strengths": ["..."], "risks": ["..."]},\n'
        '  "winner": "RAG|Baseline|Empate",\n'
        '  "summary": "comparacao curta em ate 2 frases"\n'
        "}"
    )

    message = get_llm(
        model_name=selected_judge_model,
        temperature=0.0,
        api_key=api_key,
    ).invoke([SystemMessage(content=system), HumanMessage(content=human)])
    payload = _extract_json(str(message.content or ""))
    metadata = message.response_metadata or {}
    normalized = _normalize_payload(payload, selected_judge_model)
    normalized["effective_judge_model"] = (
        metadata.get("model_name") or selected_judge_model
    )
    normalized["token_usage"] = metadata.get("token_usage")
    return normalized


def _average_score(results: list[dict], key: str) -> float:
    scores = [result[key]["score"] for result in results if key in result]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 1)


def _winner_from_scores(rag_score: float, baseline_score: float) -> str:
    if abs(rag_score - baseline_score) < 0.1:
        return "Empate"
    return "RAG" if rag_score > baseline_score else "Baseline"


def evaluate_answers_with_judges(
    question: str,
    rag_result: dict,
    baseline_result: dict,
    judge_models: list[str],
    api_key: str | None = None,
    target_successes: int = 3,
) -> dict:
    """Executa 3 avaliações independentes e retorna a média das notas."""

    results = []
    failures = []
    for judge_model in judge_models:
        if len(results) >= target_successes:
            break
        try:
            results.append(
                evaluate_answers(
                    question=question,
                    rag_result=rag_result,
                    baseline_result=baseline_result,
                    judge_model=judge_model,
                    api_key=api_key,
                )
            )
        except Exception as exc:
            failures.append({"judge_model": judge_model, "error": str(exc)})

    if not results:
        raise RuntimeError(
            "Nenhum avaliador conseguiu concluir a avaliação de qualidade."
        )

    rag_score = _average_score(results, "rag")
    baseline_score = _average_score(results, "baseline")
    winner = _winner_from_scores(rag_score, baseline_score)
    summaries = [result.get("summary", "") for result in results if result.get("summary")]
    summary = (
        f"Média de {len(results)} avaliador(es): RAG {rag_score}/5 vs "
        f"Baseline {baseline_score}/5. Melhor resultado: {winner}."
    )
    if summaries:
        summary += f" {summaries[0]}"

    return {
        "rag": {
            "score": rag_score,
            "justification": (
                f"Nota média calculada a partir de {len(results)} avaliações independentes."
            ),
            "strengths": [],
            "risks": [],
        },
        "baseline": {
            "score": baseline_score,
            "justification": (
                f"Nota média calculada a partir de {len(results)} avaliações independentes."
            ),
            "strengths": [],
            "risks": [],
        },
        "winner": winner,
        "summary": summary,
        "judge_models": judge_models,
        "judge_results": results,
        "failed_judges": failures,
    }
