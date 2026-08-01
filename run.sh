#!/usr/bin/env bash
# =============================================================================
# run.sh — Wrapper simples sobre o Docker Compose.
# Tudo roda dentro do Docker (sem venv local).
#
#   ./run.sh            -> sobe tudo (Postgres + app + ingestão + Streamlit)
#   ./run.sh down       -> para e remove os containers
#   ./run.sh ingest     -> força reindexação do CDC
#   ./run.sh eval       -> roda a avaliação RAGAS (RAG vs baseline)
#   ./run.sh logs       -> acompanha os logs
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERRO: Docker nao encontrado. Instale o Docker e o docker compose."
  exit 1
fi

CMD="${1:-up}"

case "$CMD" in
  up)
    echo "Subindo o projeto com Docker Compose..."
    docker compose up --build
    ;;
  down)
    echo "Parando containers..."
    docker compose down
    ;;
  ingest)
    echo "Reindexando CDC..."
    docker compose run --rm -e RUN_MODE=ingest app
    ;;
  eval)
    echo "Rodando avaliacao RAGAS..."
    docker compose run --rm -e RUN_MODE=eval app python -m src.evaluation
    ;;
  logs)
    docker compose logs -f
    ;;
  *)
    echo "Uso: ./run.sh [up|down|ingest|eval|logs]"
    exit 1
    ;;
esac
