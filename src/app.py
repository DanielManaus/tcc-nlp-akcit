"""Interface web Streamlit do chatbot RAG-CDC (POC)."""

import streamlit as st

from src.config import (
    OPENROUTER_MODEL,
    has_llm_credentials,
    openrouter_api_key_options,
)
from src.models import list_free_models
from src.rag_chain import answer_question


st.set_page_config(
    page_title="Chatbot RAG-CDC (POC)",
    page_icon="⚖️",
    layout="wide",
)

st.title("Chatbot RAG — Código de Defesa do Consumidor")
st.caption(
    f"POC de TCC · OpenRouter: `{OPENROUTER_MODEL}` · comparação RAG vs baseline"
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


def model_error_message(exc: Exception) -> str:
    detail = str(exc)
    if "free-models-per-day" in detail or "Rate limit" in detail or "429" in detail:
        return (
            "⚠️ O limite diário dos modelos gratuitos do OpenRouter foi atingido.\n\n"
            "Para continuar hoje, escolha uma destas opções:\n\n"
            "- aguardar o reset diário da cota gratuita;\n"
            "- trocar para outra chave API com cota disponível;\n"
            "- adicionar créditos no OpenRouter para aumentar o limite dos modelos free.\n\n"
            "O RAG e o banco continuam funcionando; apenas a geração da LLM foi bloqueada."
        )

    return (
        "❌ Não consegui gerar com o modelo selecionado.\n\n"
        "Escolha outro modelo gratuito na barra lateral e envie a pergunta novamente.\n\n"
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


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuração")

    mode = st.radio(
        "Modo de geração",
        ["RAG (com recuperação)", "Baseline (sem recuperação)"],
        help="RAG recupera artigos do CDC; Baseline usa só o conhecimento do modelo.",
    )

    compare_mode = st.checkbox(
        "Comparar RAG x Baseline",
        value=False,
        help="Responde a mesma pergunta nos dois modos. Ideal para a banca.",
    )

    api_key_options = openrouter_api_key_options()
    api_key_labels = [option["label"] for option in api_key_options]
    selected_api_key_label = st.selectbox(
        "Chave OpenRouter",
        api_key_labels,
        index=0,
        help=(
            "Selecione a chave numerada que será usada na próxima pergunta. "
            "O valor da chave não é exibido na tela."
        ),
    )
    selected_api_key = next(
        option["api_key"]
        for option in api_key_options
        if option["label"] == selected_api_key_label
    )

    free_models = cached_free_models()
    model_ids = [model["id"] for model in free_models]
    model_labels = {
        model["id"]: (
            f"{model['id']} · contexto {model.get('context_length') or '?'}"
            f" · {model.get('note', '')}"
        ).strip()
        for model in free_models
    }
    default_model = OPENROUTER_MODEL if OPENROUTER_MODEL in model_ids else model_ids[0]

    selected_model = st.selectbox(
        "Top modelos gratuitos OpenRouter",
        model_ids,
        index=model_ids.index(default_model),
        format_func=lambda model_id: model_labels.get(model_id, model_id),
        help="Se um modelo gratuito falhar, escolha outro e envie a pergunta novamente.",
    )

    if st.button("Atualizar lista de modelos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "Use RAG para a demonstração principal. Para comparação, marque "
        "'Comparar RAG x Baseline'. Mensagens antigas mantêm o modo usado na época."
    )

    active_mode = "Comparação RAG x Baseline" if compare_mode else (
        "RAG" if mode.startswith("RAG") else "Baseline"
    )
    st.caption("Próxima pergunta")
    st.code(active_mode, language=None)
    st.caption("Modelo selecionado")
    st.code(selected_model, language=None)
    st.caption("Chave selecionada")
    st.code(selected_api_key_label, language=None)

examples = [
    "Quais são os direitos básicos do consumidor?",
    "Como funciona o direito de arrependimento?",
    "Teste de recusa: o CDC fala sobre guarda compartilhada?",
]

selected_example = None
cols = st.columns(3)
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        selected_example = example.replace("Teste de recusa: ", "")

st.caption(f"Modo ativo para a próxima pergunta: `{active_mode}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("generation_mode"):
            st.caption(f"Modo usado: `{msg['generation_mode']}`")
        if msg.get("api_key_label"):
            st.caption(f"Chave usada: `{msg['api_key_label']}`")
        if msg.get("effective_model"):
            st.caption(f"Modelo usado: `{msg['effective_model']}`")

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
            st.caption(f"Chave usada: `{selected_api_key_label}`")
            st.caption(f"Modelo usado: `{rag.get('effective_model') or selected_model}`")
            if not rag.get("error"):
                render_sources(
                    rag.get("documents", []),
                    rag.get("history_documents", []),
                    rag.get("jurisprudence_documents", []),
                )

            st.markdown("---")
            st.markdown("### Baseline (sem recuperação)")
            st.markdown(baseline["answer"])
            st.caption(f"Chave usada: `{selected_api_key_label}`")
            st.caption(
                f"Modelo usado: `{baseline.get('effective_model') or selected_model}`"
            )

        combined_answer = (
            "### RAG (com recuperação)\n"
            f"{rag['answer']}\n\n"
            "### Baseline (sem recuperação)\n"
            f"{baseline['answer']}"
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
            st.caption(f"Modo usado: `{generation_mode}`")
            st.caption(f"Chave usada: `{selected_api_key_label}`")
            if result.get("effective_model"):
                st.caption(f"Modelo usado: `{result['effective_model']}`")
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
