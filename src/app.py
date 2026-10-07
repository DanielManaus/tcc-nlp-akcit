"""Interface web Streamlit do chatbot RAG-CDC (POC)."""

import streamlit as st

from src.config import (
    OPENROUTER_MODEL,
    has_llm_credentials,
    openrouter_api_key_options,
)
from src.models import list_free_models
from src.quality_judge import evaluate_answers_with_judges
from src.rag_chain import answer_question
from src.history_store import (
    history_stats,
    list_interactions,
    save_interaction,
    update_human_review,
)


st.set_page_config(
    page_title="RAG-CDC | TCC NLP",
    page_icon="⚖️",
    layout="wide",
)

st.title("Chatbot RAG — Código de Defesa do Consumidor")
st.caption(
    "Projeto de TCC desenvolvido pela equipe · RAG aplicado ao Código de Defesa do Consumidor"
)

QUALITY_JUDGE_PRIORITY = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash",
    "amazon/nova-lite-v1",
]

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
    judge_models: list[str],
    api_key: str,
) -> dict:
    try:
        return evaluate_answers_with_judges(
            question=question,
            rag_result=rag_result,
            baseline_result=baseline_result,
            judge_models=judge_models,
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

    with st.expander("Notas individuais dos avaliadores"):
        rows = []
        for result in evaluation.get("judge_results", []):
            rows.append(
                {
                    "Avaliador": result.get("judge_model_display")
                    or result.get("effective_judge_model")
                    or result.get("judge_model"),
                    "RAG": result.get("rag", {}).get("score"),
                    "Baseline": result.get("baseline", {}).get("score"),
                    "Melhor": result.get("winner"),
                }
            )
        if rows:
            st.table(rows)
        for failure in evaluation.get("failed_judges", []):
            st.warning(f"Avaliador indisponível: `{failure['judge_model']}`")


def format_quality_markdown(evaluation: dict) -> str:
    if not evaluation or evaluation.get("error"):
        return ""

    rag_eval = evaluation.get("rag", {})
    baseline_eval = evaluation.get("baseline", {})
    judge_names = evaluation.get("judge_models_display") or []
    judge_line = (
        f"- Avaliadores: {', '.join(judge_names)}\n" if judge_names else ""
    )
    return (
        "### Avaliação de qualidade\n"
        f"- Nota RAG: {rag_eval.get('score', 0)}/5\n"
        f"- Nota Baseline: {baseline_eval.get('score', 0)}/5\n"
        f"- Melhor resposta: {evaluation.get('winner', 'Indefinido')}\n"
        f"{judge_line}"
        f"- Resumo: {evaluation.get('summary', '')}"
    )


def select_quality_judges(selected_model: str, models: list[dict]) -> list[str]:
    paid_model_ids = {
        model["id"]
        for model in models
        if (model.get("tier") or "").lower() == "pago"
    }
    ordered_paid_models = [
        model_id for model_id in QUALITY_JUDGE_PRIORITY if model_id in paid_model_ids
    ]
    ordered_paid_models.extend(
        model_id
        for model_id in paid_model_ids
        if model_id not in ordered_paid_models
    )
    candidates = [
        model_id for model_id in ordered_paid_models if model_id != selected_model
    ]
    if len(candidates) < 3:
        candidates = ordered_paid_models
    return candidates[:3]


def score_label(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}/5"


def created_at_label(value) -> str:
    if not value:
        return ""
    try:
        return value.strftime("%d/%m/%Y %H:%M")
    except AttributeError:
        return str(value)


def save_interaction_safely(**kwargs) -> int | None:
    try:
        return save_interaction(**kwargs)
    except Exception as exc:
        st.warning(
            "A resposta foi gerada, mas não consegui salvar no histórico. "
            f"Detalhe técnico: `{exc}`"
        )
        return None


def render_history_tab() -> None:
    st.subheader("Histórico e avaliação humana")
    st.caption(
        "Registros persistidos no PostgreSQL da POC. A nota humana permite "
        "comparar a avaliação automática com a percepção da equipe/orientador."
    )

    try:
        stats = history_stats()
    except Exception as exc:
        st.warning(f"Não consegui carregar o histórico: `{exc}`")
        return

    total, avg_human, avg_ai_rag, pending = st.columns(4)
    total.metric("Perguntas salvas", int(stats.get("total") or 0))
    avg_human.metric("Média humana", score_label(stats.get("avg_human")))
    avg_ai_rag.metric("Média RAG automática", score_label(stats.get("avg_ai_rag")))
    pending.metric("Pendentes de nota", int(stats.get("pending_human") or 0))

    search_col, pending_col, limit_col = st.columns([2, 1, 1])
    with search_col:
        search = st.text_input(
            "Buscar no histórico",
            placeholder="Ex.: arrependimento, produto com defeito, banco...",
            help="Quando possível, a busca usa o embedding da pergunta no pgvector.",
        )
    with pending_col:
        only_pending = st.checkbox("Só pendentes", value=False)
    with limit_col:
        limit = st.selectbox("Quantidade", [10, 20, 30, 50], index=1)

    try:
        rows = list_interactions(
            limit=limit,
            search=search,
            only_pending_human_review=only_pending,
        )
    except Exception as exc:
        st.warning(f"Não consegui consultar o histórico: `{exc}`")
        return

    if not rows:
        st.info("Nenhum registro encontrado para os filtros atuais.")
        return

    for row in rows:
        human_badge = (
            f"nota humana {score_label(row.get('human_score'))}"
            if row.get("human_score") is not None
            else "aguardando nota humana"
        )
        title = (
            f"#{row['id']} · {created_at_label(row.get('created_at'))} · "
            f"{row.get('mode')} · {human_badge}"
        )
        with st.expander(title):
            st.markdown("**Pergunta**")
            st.write(row.get("question", ""))

            info_cols = st.columns(4)
            info_cols[0].metric("Nota humana", score_label(row.get("human_score")))
            info_cols[1].metric("Nota RAG", score_label(row.get("ai_rag_score")))
            info_cols[2].metric(
                "Nota Baseline",
                score_label(row.get("ai_baseline_score")),
            )
            info_cols[3].metric("Melhor automática", row.get("ai_winner") or "—")

            st.caption(
                "Modelo: "
                f"`{row.get('response_model_label') or row.get('response_model') or '—'}`"
            )

            if row.get("compare_mode"):
                rag_tab, baseline_tab, quality_tab = st.tabs(
                    ["Resposta RAG", "Resposta Baseline", "Avaliação automática"]
                )
                with rag_tab:
                    st.markdown(row.get("rag_answer") or "Sem resposta RAG.")
                with baseline_tab:
                    st.markdown(
                        row.get("baseline_answer") or "Sem resposta baseline."
                    )
                with quality_tab:
                    quality = row.get("ai_quality") or {}
                    if quality:
                        st.info(quality.get("summary") or "Avaliação salva.")
                        rows_quality = []
                        for result in quality.get("judge_results", []):
                            rows_quality.append(
                                {
                                    "Avaliador": result.get("judge_model_display")
                                    or result.get("effective_judge_model")
                                    or result.get("judge_model"),
                                    "RAG": result.get("rag", {}).get("score"),
                                    "Baseline": result.get("baseline", {}).get(
                                        "score"
                                    ),
                                    "Melhor": result.get("winner"),
                                }
                            )
                        if rows_quality:
                            st.table(rows_quality)
                    else:
                        st.caption("Sem avaliação automática registrada.")
            else:
                st.markdown("**Resposta**")
                st.markdown(
                    row.get("single_answer")
                    or row.get("rag_answer")
                    or row.get("baseline_answer")
                    or "Sem resposta registrada."
                )

            counts = row.get("source_counts") or {}
            if counts:
                st.caption(
                    "Fontes recuperadas: "
                    f"CDC `{counts.get('cdc', 0)}` · "
                    f"Histórico `{counts.get('historico', 0)}` · "
                    f"STJ `{counts.get('stj', 0)}`"
                )

            with st.form(f"human_review_{row['id']}"):
                st.markdown("**Avaliação humana**")
                current_score = (
                    float(row["human_score"])
                    if row.get("human_score") is not None
                    else 4.0
                )
                score = st.slider(
                    "Nota da resposta principal",
                    min_value=0.0,
                    max_value=5.0,
                    value=current_score,
                    step=0.5,
                    help=(
                        "Na comparação, avalie a resposta RAG como resposta "
                        "principal da POC."
                    ),
                )
                reviewer = st.text_input(
                    "Avaliador",
                    value=row.get("human_reviewer") or "",
                    placeholder="Ex.: orientador, equipe, avaliador 1",
                )
                comment = st.text_area(
                    "Comentário",
                    value=row.get("human_comment") or "",
                    placeholder="O que ficou bom? O que precisa melhorar?",
                    height=90,
                )
                submitted = st.form_submit_button(
                    "Salvar avaliação humana",
                    use_container_width=True,
                )
                if submitted:
                    update_human_review(
                        int(row["id"]),
                        human_score=score,
                        human_comment=comment,
                        human_reviewer=reviewer,
                    )
                    st.success("Avaliação humana salva.")
                    st.rerun()


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
        help=(
            "No modo comparação, três modelos pagos baratos avaliam as respostas "
            "e a nota final é a média."
        ),
    )
    judge_models = select_quality_judges(selected_model, free_models)

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
            st.caption("Avaliadores de qualidade")
            for judge_model in judge_models:
                st.markdown(f"- **{model_caption(judge_model)}**")
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

demo_tab, history_tab = st.tabs(["Demonstração", "Histórico"])

with demo_tab:
    status_caption = f"Demonstração ativa: `{active_mode}`"
    if evaluate_quality:
        status_caption += " · `média de 3 avaliadores`"
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
            if msg.get("interaction_id"):
                details.append(f"Histórico: `#{msg['interaction_id']}`")
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
                            judge_models=judge_models,
                            api_key=selected_api_key,
                        )
                        for result in quality.get("judge_results", []):
                            result["judge_model_display"] = model_caption(
                                result.get("effective_judge_model")
                                or result.get("judge_model")
                            )
                        quality["judge_models_display"] = [
                            model_caption(model_id) for model_id in judge_models
                        ]
                    render_quality_evaluation(quality)
                elif evaluate_quality:
                    st.warning(
                        "A avaliação de qualidade foi pulada porque uma das respostas "
                        "teve erro de geração."
                    )

                interaction_id = save_interaction_safely(
                    question=prompt,
                    mode=active_mode,
                    compare_mode=True,
                    response_model=selected_model,
                    response_model_label=model_caption(selected_model),
                    api_key_label=selected_api_key_label,
                    rag_result=rag,
                    baseline_result=baseline,
                    ai_quality=quality,
                )
                if interaction_id:
                    st.caption(f"Registro salvo no histórico: `#{interaction_id}`")

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
                    "interaction_id": interaction_id,
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

                interaction_id = save_interaction_safely(
                    question=prompt,
                    mode=generation_mode,
                    compare_mode=False,
                    response_model=selected_model,
                    response_model_label=model_caption(selected_model),
                    api_key_label=selected_api_key_label,
                    single_result=result,
                )
                if interaction_id:
                    st.caption(f"Registro salvo no histórico: `#{interaction_id}`")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "generation_mode": generation_mode,
                    "effective_model": result.get("effective_model"),
                    "api_key_label": selected_api_key_label,
                    "interaction_id": interaction_id,
                }
            )

with history_tab:
    render_history_tab()
