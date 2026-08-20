"""Interface web Streamlit do chatbot RAG-CDC (POC)."""

import streamlit as st

from src.config import (
    OPENROUTER_MODEL,
    has_llm_credentials,
    openrouter_api_key_options,
)
from src.models import list_free_models
from src.quality_judge import DEFAULT_JUDGE_MODEL, evaluate_answers
from src.rag_chain import answer_question


st.set_page_config(
    page_title="RAG-CDC | TCC NLP",
    page_icon="⚖️",
    layout="wide",
)

st.title("Chatbot RAG — Código de Defesa do Consumidor")
st.caption(
    "Projeto de TCC desenvolvido pela equipe · RAG aplicado ao Código de Defesa do Consumidor"
)

if not has_llm_credentials():
    st.error(
        "⚠️ Nenhuma chave OpenRouter definida. Crie um arquivo `.env` "
        "a partir de `.env.example` e informe ao menos uma chave em "
        "`OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2` ou "
        "`OPENROUTER_API_KEY_3`."
    )
    st.stop()


@st.cache_data(ttl=900, show_spinner=False)
def cached_free_models() -> list[dict]:
    return list_free_models()


def format_context_length(context_length: int | None) -> str:
    if not context_length:
        return "ctx ?"
    if context_length >= 1_000_000:
        return "ctx 1M"
    if context_length >= 1_000:
        return f"ctx {round(context_length / 1000)}K"
    return f"ctx {context_length}"


def format_model_label(model: dict) -> str:
    tier = model.get("tier") or (
        "Grátis" if str(model.get("id", "")).endswith(":free") else "Pago"
    )
    display_name = model.get("display_name") or model.get("name") or model["id"]
    parameters = model.get("parameters") or "não divulgado"
    if parameters == "não divulgado":
        parameters = "parâmetros não divulgados"
    return (
        f"{tier} · {display_name} · {parameters} · "
        f"{format_context_length(model.get('context_length'))}"
    )


def model_error_message(exc: Exception) -> str:
    detail = str(exc)
    if "free-models-per-day" in detail or "Rate limit" in detail or "429" in detail:
        return (
            "⚠️ O limite diário dos modelos gratuitos do OpenRouter foi atingido.\n\n"
            "Para continuar hoje, escolha uma destas opções:\n\n"
            "- aguardar o reset diário da cota gratuita;\n"
            "- trocar para outra chave API com cota disponível;\n"
            "- adicionar créditos no OpenRouter para aumentar o limite dos modelos free.\n\n"
            "O RAG e o banco continuam funcionando; apenas a geração da resposta foi bloqueada."
        )

    return (
        "❌ Não consegui gerar com o modelo selecionado.\n\n"
        "Escolha outro modelo na barra lateral e envie a pergunta novamente.\n\n"
        f"Detalhe técnico: `{detail}`"
    )


def render_sources(docs: list, history_docs: list, stj_docs: list) -> None:
    with st.expander("Trechos legais recuperados do CDC"):
        if not docs:
            st.caption("Nenhum trecho legal recuperado.")
        for i, doc in enumerate(docs, start=1):
            artigo = doc.metadata.get("artigo", "N/A")
            trecho = doc.page_content[:900].strip()
            st.markdown(f"**{i}. {artigo}**")
            st.caption(trecho)

    with st.expander("Contexto histórico do CDC"):
        if not history_docs:
            st.caption("Nenhum contexto histórico recuperado para esta pergunta.")
        for i, doc in enumerate(history_docs, start=1):
            referencia = doc.metadata.get("referencia", "Histórico CDC")
            tema = doc.metadata.get("tema", "")
            trecho = doc.page_content[:900].strip()
            st.markdown(f"**{i}. {referencia}**")
            if tema:
                st.caption(f"Tema: {tema}")
            st.caption(trecho)

    with st.expander("Jurisprudência complementar do STJ"):
        if not stj_docs:
            st.caption("Nenhuma súmula complementar recuperada para esta pergunta.")
        for i, doc in enumerate(stj_docs, start=1):
            referencia = doc.metadata.get("referencia", "STJ")
            tema = doc.metadata.get("tema", "")
            trecho = doc.page_content[:900].strip()
            st.markdown(f"**{i}. {referencia}**")
            if tema:
                st.caption(f"Tema: {tema}")
            st.caption(trecho)


def run_generation(
    question: str,
    use_rag: bool,
    model_name: str,
    api_key: str,
) -> dict:
    try:
        return answer_question(
            question,
            use_rag=use_rag,
            model_name=model_name,
            api_key=api_key,
        )
    except Exception as exc:
        return {
            "answer": model_error_message(exc),
            "effective_model": None,
            "documents": [],
            "history_documents": [],
            "jurisprudence_documents": [],
            "error": True,
        }


def run_quality_evaluation(
    question: str,
    rag_result: dict,
    baseline_result: dict,
    judge_model: str,
    api_key: str,
) -> dict:
    try:
        return evaluate_answers(
            question=question,
            rag_result=rag_result,
            baseline_result=baseline_result,
            judge_model=judge_model,
            api_key=api_key,
        )
    except Exception as exc:
        return {
            "error": True,
            "message": (
                "⚠️ Não consegui executar a avaliação automática de qualidade.\n\n"
                f"Detalhe técnico: `{str(exc)}`"
            ),
        }


def _render_list(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def render_quality_evaluation(evaluation: dict) -> None:
    st.markdown("---")
    st.markdown("### Avaliação de qualidade")

    if evaluation.get("error"):
        st.warning(evaluation["message"])
        return

    rag_eval = evaluation.get("rag", {})
    baseline_eval = evaluation.get("baseline", {})

    col_rag, col_baseline, col_winner = st.columns([1, 1, 1])
    with col_rag:
        st.metric("Nota RAG", f"{rag_eval.get('score', 0)}/5")
    with col_baseline:
        st.metric("Nota Baseline", f"{baseline_eval.get('score', 0)}/5")
    with col_winner:
        st.metric("Melhor resposta", evaluation.get("winner", "Indefinido"))

    if evaluation.get("summary"):
        st.info(evaluation["summary"])

    with st.expander("Detalhes da avaliação"):
        st.markdown("**RAG**")
        st.caption(rag_eval.get("justification", "Sem justificativa."))
        if rag_eval.get("strengths"):
            st.markdown("Pontos fortes:")
            _render_list(rag_eval["strengths"])
        if rag_eval.get("risks"):
            st.markdown("Riscos/limitações:")
            _render_list(rag_eval["risks"])

        st.markdown("**Baseline**")
        st.caption(baseline_eval.get("justification", "Sem justificativa."))
        if baseline_eval.get("strengths"):
            st.markdown("Pontos fortes:")
            _render_list(baseline_eval["strengths"])
        if baseline_eval.get("risks"):
            st.markdown("Riscos/limitações:")
            _render_list(baseline_eval["risks"])

    st.caption(
        "Avaliador: "
        f"`{evaluation.get('judge_model_display') or evaluation.get('effective_judge_model') or evaluation.get('judge_model')}`"
    )


def format_quality_markdown(evaluation: dict) -> str:
    if not evaluation or evaluation.get("error"):
        return ""

    rag_eval = evaluation.get("rag", {})
    baseline_eval = evaluation.get("baseline", {})
    return (
        "### Avaliação de qualidade\n"
        f"- Nota RAG: {rag_eval.get('score', 0)}/5\n"
        f"- Nota Baseline: {baseline_eval.get('score', 0)}/5\n"
        f"- Melhor resposta: {evaluation.get('winner', 'Indefinido')}\n"
        f"- Resumo: {evaluation.get('summary', '')}\n"
        f"- Avaliador: "
        f"{evaluation.get('judge_model_display') or evaluation.get('effective_judge_model') or evaluation.get('judge_model')}"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Demonstração")

    mode = st.radio(
        "Tipo de resposta",
        ["RAG (com fontes)", "Baseline (sem fontes)"],
        help="RAG consulta a base do CDC; Baseline responde sem recuperação.",
    )

    compare_mode = st.checkbox(
        "Comparar RAG x Baseline",
        value=False,
        help="Responde a mesma pergunta nos dois modos.",
    )

    api_key_options = openrouter_api_key_options()
    api_key_labels = [option["label"] for option in api_key_options]
    selected_api_key_label = st.selectbox(
        "Chave da equipe",
        api_key_labels,
        index=0,
        help="Seleciona a chave usada na demonstração, sem exibir o valor.",
    )
    selected_api_key = next(
        option["api_key"]
        for option in api_key_options
        if option["label"] == selected_api_key_label
    )

    free_models = cached_free_models()
    model_ids = [model["id"] for model in free_models]
    model_labels = {model["id"]: format_model_label(model) for model in free_models}

    def model_caption(model_id: str | None) -> str:
        if not model_id:
            return "Modelo não identificado"
        return model_labels.get(model_id, model_id)

    default_model = OPENROUTER_MODEL if OPENROUTER_MODEL in model_ids else model_ids[0]

    selected_model = st.selectbox(
        "Modelo de resposta",
        model_ids,
        index=model_ids.index(default_model),
        format_func=lambda model_id: model_labels.get(model_id, model_id),
        help="Modelo usado para responder às perguntas.",
    )

    evaluate_quality = st.checkbox(
        "Avaliação de qualidade",
        value=compare_mode,
        disabled=not compare_mode,
        help="No modo comparação, atribui nota de 0 a 5 para cada resposta.",
    )

    judge_model = selected_model
    if evaluate_quality:
        judge_default = (
            DEFAULT_JUDGE_MODEL
            if DEFAULT_JUDGE_MODEL in model_ids
            else selected_model
        )
        judge_model = st.selectbox(
            "Avaliador",
            model_ids,
            index=model_ids.index(judge_default),
            format_func=lambda model_id: model_labels.get(model_id, model_id),
            help="Usado apenas para avaliar a qualidade das respostas.",
        )

    if st.button("Atualizar modelos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    active_mode = "Comparação RAG x Baseline" if compare_mode else (
        "RAG" if mode.startswith("RAG") else "Baseline"
    )

    with st.expander("Detalhes técnicos", expanded=False):
        st.caption("Modo")
        st.markdown(f"**{active_mode}**")
        st.caption("Modelo de resposta")
        st.markdown(f"**{model_caption(selected_model)}**")
        if evaluate_quality:
            st.caption("Avaliador")
            st.markdown(f"**{model_caption(judge_model)}**")
        st.caption("Chave")
        st.markdown(f"**{selected_api_key_label}**")

examples = [
    {
        "title": "Arrependimento",
        "preview": "Compra online chegou, mas o consumidor quer desistir.",
        "question": (
            "Comprei um produto pela internet, ele chegou na minha casa, mas eu "
            "me arrependi da compra. Em quais situações posso desistir, qual é "
            "o prazo para fazer isso e o que o fornecedor precisa devolver?"
        ),
    },
    {
        "title": "Produto com defeito",
        "preview": "Produto apresentou problema e a loja não resolveu.",
        "question": (
            "Comprei um produto que apresentou defeito poucos dias depois do uso. "
            "A loja disse que vai mandar para assistência, mas já passou bastante "
            "tempo e o problema não foi resolvido. Quais opções o CDC dá ao "
            "consumidor se o vício não for sanado no prazo legal?"
        ),
    },
    {
        "title": "Banco e fraude",
        "preview": "Transação bancária suspeita ou golpe.",
        "question": (
            "Percebi uma transação bancária que não reconheço, possivelmente "
            "relacionada a golpe ou fraude. O CDC se aplica a bancos? E como "
            "fica a responsabilidade da instituição financeira nesses casos?"
        ),
    },
    {
        "title": "Recusa fora do CDC",
        "preview": "Assunto fora do CDC para testar recusa correta.",
        "question": (
            "Tenho uma dúvida sobre guarda compartilhada de filhos após separação. "
            "O Código de Defesa do Consumidor trata desse assunto ou isso está "
            "fora do escopo do CDC?"
        ),
    },
]

status_caption = f"Demonstração ativa: `{active_mode}`"
if evaluate_quality:
    status_caption += " · `avaliação de qualidade habilitada`"
st.caption(status_caption)

selected_example = None
st.caption("Cenários para demonstrar RAG x Baseline:")
cols = st.columns(4)
for index, (col, example) in enumerate(zip(cols, examples)):
    col.markdown(f"**{example['title']}**")
    col.caption(example["preview"])
    if col.button("Usar pergunta", key=f"example_{index}", use_container_width=True):
        selected_example = example["question"]

st.caption(f"Modo ativo para a próxima pergunta: `{active_mode}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        details = []
        if msg.get("generation_mode"):
            details.append(f"Modo: `{msg['generation_mode']}`")
        if msg.get("effective_model"):
            details.append(f"Modelo: `{model_caption(msg['effective_model'])}`")
        if details:
            st.caption(" · ".join(details))

prompt = selected_example or st.chat_input("Faça uma pergunta sobre o CDC...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if compare_mode:
        with st.chat_message("assistant"):
            with st.spinner("Consultando RAG e baseline..."):
                rag = run_generation(
                    prompt,
                    use_rag=True,
                    model_name=selected_model,
                    api_key=selected_api_key,
                )
                baseline = run_generation(
                    prompt,
                    use_rag=False,
                    model_name=selected_model,
                    api_key=selected_api_key,
                )

            st.markdown("### RAG (com recuperação)")
            st.markdown(rag["answer"])
            st.caption(
                f"Modo: `RAG` · Modelo: `{model_caption(rag.get('effective_model') or selected_model)}`"
            )
            if not rag.get("error"):
                render_sources(
                    rag.get("documents", []),
                    rag.get("history_documents", []),
                    rag.get("jurisprudence_documents", []),
                )

            st.markdown("---")
            st.markdown("### Baseline (sem recuperação)")
            st.markdown(baseline["answer"])
            st.caption(
                f"Modo: `Baseline` · Modelo: `{model_caption(baseline.get('effective_model') or selected_model)}`"
            )

            quality = None
            if evaluate_quality and not rag.get("error") and not baseline.get("error"):
                with st.spinner("Executando avaliação de qualidade..."):
                    quality = run_quality_evaluation(
                        question=prompt,
                        rag_result=rag,
                        baseline_result=baseline,
                        judge_model=judge_model,
                        api_key=selected_api_key,
                    )
                    quality["judge_model_display"] = model_caption(
                        quality.get("effective_judge_model") or judge_model
                    )
                render_quality_evaluation(quality)
            elif evaluate_quality:
                st.warning(
                    "A avaliação de qualidade foi pulada porque uma das respostas "
                    "teve erro de geração."
                )

        quality_markdown = format_quality_markdown(quality) if quality else ""
        combined_answer = (
            "### RAG (com recuperação)\n"
            f"{rag['answer']}\n\n"
            "### Baseline (sem recuperação)\n"
            f"{baseline['answer']}\n\n"
            f"{quality_markdown}"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": combined_answer,
                "generation_mode": "Comparação RAG x Baseline",
                "effective_model": selected_model,
                "api_key_label": selected_api_key_label,
            }
        )
    else:
        generation_mode = "RAG" if mode.startswith("RAG") else "Baseline"
        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                result = run_generation(
                    prompt,
                    use_rag=generation_mode == "RAG",
                    model_name=selected_model,
                    api_key=selected_api_key,
                )

            answer = result["answer"]
            st.markdown(answer)
            response_details = [f"Modo: `{generation_mode}`"]
            if result.get("effective_model"):
                response_details.append(
                    f"Modelo: `{model_caption(result['effective_model'])}`"
                )
            st.caption(" · ".join(response_details))
            if generation_mode == "RAG" and not result.get("error"):
                render_sources(
                    result.get("documents", []),
                    result.get("history_documents", []),
                    result.get("jurisprudence_documents", []),
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "generation_mode": generation_mode,
                "effective_model": result.get("effective_model"),
                "api_key_label": selected_api_key_label,
            }
        )
