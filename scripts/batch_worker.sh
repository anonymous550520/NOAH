#!/usr/bin/env bash
export OPENAI_API_KEY="your_openai_api_key"
# Wrapper to run batch worker once or in watch mode
# Usage:
#   ./scripts/batch_worker.sh

ROOT_DIR="outputs/batch_runs"
MODE="watch"

if [[ "$MODE" == "once" ]]; then
  python -m src.evaluate.batch_worker --root "$ROOT_DIR" --once
elif [[ "$MODE" == "watch" ]]; then
  python -m src.evaluate.batch_worker --root "$ROOT_DIR"
else
  # Treat as batch id
  python -m src.evaluate.batch_worker --root "$ROOT_DIR" --batch-id "$MODE" --once
fi
