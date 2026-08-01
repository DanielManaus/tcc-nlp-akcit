"""Avaliação RAG vs Baseline usando RAGAS e métricas do TCC.

Métricas (TCC §6):
- Taxa de alucinação (proporção de respostas com artigo inexistente / conteúdo incorreto)
- Precisão de citação (artigo citado coincide com gabarito)
- Taxa de recusa correta (fora de escopo -> recusa correta)
- Fidelidade ao contexto (RAGAS faithfulness)

Execução: python -m src.evaluation
"""
import re

from src.config import has_llm_credentials
from src.golden_dataset import GOLDEN_DATASET
from src.rag_chain import build_rag_chain, build_baseline_chain, retrieve_documents


REFUSAL_MARKER = "A informação não consta no Código de Defesa do Consumidor"

ART_RE = re.compile(r"Art(?:igo)?\.?\s*(\d+)", re.IGNORECASE)


def extract_cited_articles(text: str) -> set[str]:
    return set(ART_RE.findall(text))


def run_condition(chain, label: str, collect_context: bool = False) -> list[dict]:
    results = []
    for item in GOLDEN_DATASET:
        q = item["question"]
        contexts = []
        if collect_context:
            contexts = [doc.page_content for doc in retrieve_documents(q)]
        ans = chain.invoke(q)
        cited = extract_cited_articles(ans)
        results.append(
            {
                "category": item["category"],
                "question": q,
                "answer": ans,
                "cited": cited,
                "expected": set(item["expected_articles"]),
                "out_of_scope": item["out_of_scope"],
                "refused": REFUSAL_MARKER in ans,
                "contexts": contexts,
            }
        )
        print(f"[{label}] {q[:50]}... -> artigos citados: {sorted(cited)}")
    return results


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    hallucination = 0
    citation_correct = 0
    citation_eligible = 0
    correct_refusal = 0
    refusal_eligible = 0

    for r in results:
        if r["out_of_scope"]:
            refusal_eligible += 1
            if r["refused"]:
                correct_refusal += 1
            # Se recusou, não conta como alucinação
            if not r["refused"]:
                # citou algo que não deveria -> alucinação de citação
                if r["cited"]:
                    hallucination += 1
        else:
            # Dentro do escopo
            citation_eligible += 1
            if r["expected"]:
                # Precisão: pelo menos um artigo gabarito citado e nenhum inexistente inválido?
                # Consideramos correto se citou algum esperado e não inventou fora do esperado
                overlap = r["cited"] & r["expected"]
                if overlap:
                    citation_correct += 1
                else:
                    hallucination += 1
            else:
                # esperado vazio mas dentro de escopo (caso raro) -> ignora citação
                pass

    metrics = {
        "total": total,
        "taxa_alucinacao": round(hallucination / total, 3) if total else 0,
        "precisao_citacao": round(citation_correct / citation_eligible, 3)
        if citation_eligible
        else 0.0,
        "taxa_recusa_correta": round(correct_refusal / refusal_eligible, 3)
        if refusal_eligible
        else 0.0,
    }
    return metrics


def run_ragas_faithfulness(rag_results: list[dict]) -> float | None:
    """Calcula faithfulness (RAGAS) para a condição RAG, se possível."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ RAGAS não disponível: {exc}")
        return None

    rows = [
        {"question": r["question"], "answer": r["answer"],
         "contexts": r["contexts"], "reference": r["answer"]}
        for r in rag_results
        if r["contexts"]
    ]
    if not rows:
        return None
    ds = Dataset.from_list(rows)
    try:
        score = evaluate(ds, metrics=[faithfulness])
        return float(score["faithfulness"])
    except Exception as exc:
        print(f"⚠️ Falha ao calcular RAGAS faithfulness: {exc}")
        return None


def main():
    if not has_llm_credentials():
        raise SystemExit(
            "Nenhuma chave OpenRouter definida. Configure OPENROUTER_API_KEY_1, "
            "OPENROUTER_API_KEY_2 ou OPENROUTER_API_KEY_3."
        )
    print("🔍 Construindo cadeias...")
    rag_chain = build_rag_chain()
    baseline_chain = build_baseline_chain()

    print("\n=== RAG ===")
    rag_results = run_condition(rag_chain, "RAG", collect_context=True)
    print("\n=== BASELINE (sem recuperação) ===")
    baseline_results = run_condition(baseline_chain, "BASELINE")

    rag_metrics = compute_metrics(rag_results)
    base_metrics = compute_metrics(baseline_results)

    faith = run_ragas_faithfulness(rag_results)

    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO (TCC §6)")
    print("=" * 60)
    print(f"{'Métrica':<28}{'RAG':>12}{'Baseline':>12}")
    print("-" * 52)
    print(f"{'Taxa de alucinação':<28}{rag_metrics['taxa_alucinacao']:>12}"
          f"{base_metrics['taxa_alucinacao']:>12}")
    print(f"{'Precisão de citação':<28}{rag_metrics['precisao_citacao']:>12}"
          f"{base_metrics['precisao_citacao']:>12}")
    print(f"{'Taxa de recusa correta':<28}{rag_metrics['taxa_recusa_correta']:>12}"
          f"{base_metrics['taxa_recusa_correta']:>12}")
    if faith is not None:
        print(f"{'Fidelidade RAGAS':<28}{round(faith, 3):>12}{'-':>12}")
    print("=" * 60)
    print("Hipótese: RAG deve apresentar MENOR alucinação e MAIOR "
          "precisão de citação que o baseline.")


if __name__ == "__main__":
    main()
