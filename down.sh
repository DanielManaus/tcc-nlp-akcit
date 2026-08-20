#!/usr/bin/env bash
# =============================================================================
# down.sh — Para a POC RAG-CDC quando precisar liberar o Docker.
#
#   ./down.sh            -> para e remove containers da POC
#   ./down.sh --volumes  -> para containers e remove volumes (apaga banco/cache)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERRO: Docker nao encontrado. Instale o Docker e o docker compose."
  exit 1
fi

case "${1:-}" in
  "")
    echo "Parando containers da POC RAG-CDC..."
    docker compose down
    echo "OK: containers removidos."
    ;;
  --volumes|-v)
    echo "Parando containers e removendo volumes da POC RAG-CDC..."
    echo "ATENCAO: isso apaga o banco pgvector e o cache de embeddings."
    docker compose down --volumes
    echo "OK: containers e volumes removidos."
    ;;
  --help|-h)
    echo "Uso: ./down.sh [--volumes]"
    ;;
  *)
    echo "Uso: ./down.sh [--volumes]"
    exit 1
    ;;
esac
