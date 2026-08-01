#!/usr/bin/env bash
# Entrypoint do container: indexa CDC + STJ + histórico (idempotente) e sobe o Streamlit.
# Se RUN_MODE=eval, executa a avaliação RAGAS em vez da UI.
set -euo pipefail

cd /app
export PYTHONPATH=/app:${PYTHONPATH:-}

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if [ "${RUN_MODE:-app}" = "eval" ]; then
  echo "📊 Modo avaliação: rodando RAGAS (RAG vs baseline)..."
  python -m src.evaluation
  exit 0
fi

if [ "${RUN_MODE:-app}" = "ingest" ]; then
  echo "📥 Modo ingestão: reindexando CDC + STJ + histórico no pgvector..."
  python -c "from src.ingestion import ingest; ingest(force_recreate=True, refresh_corpus=True)"
  exit 0
fi

echo "📥 Indexando CDC + STJ + histórico no pgvector (se ainda não indexado)..."
python -m src.ingestion || {
  echo "❌ Falha na ingestão. Verifique a conexão com o Postgres e a rede."
  echo "   O Streamlit não será iniciado."
  exit 1
}

echo "🚀 Iniciando Streamlit em http://0.0.0.0:8501 ..."
exec streamlit run src/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
