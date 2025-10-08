#!/usr/bin/env bash

export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
MODE="joint"           # hallucination | omission | joint

## modify
# JUDGE_MODEL="gemini-2.0-flash"
JUDGE_MODEL="gpt-4o-2024-08-06"
RECIPES="path/to/metadata.json"
RESULTS="path/to/captioning_result.json"
OUTPUT="path/to/evaluation_result.json"
## modify

MAX_CONCURRENT="20"
API_MODE="sync"    # sync | batch
# Default output path if not provided

echo "[run] mode=$MODE api_mode=$API_MODE max_concurrent=$MAX_CONCURRENT"
echo "[in ] recipes=$RECIPES"
echo "[in ] results=$RESULTS"
echo "[out] output=$OUTPUT"

python -m src.evaluate.evaluate_captions \
  --mode "$MODE" \
  --judge_model "$JUDGE_MODEL" \
  --recipes "$RECIPES" \
  --results "$RESULTS" \
  --output "$OUTPUT" \
  --max_concurrent "$MAX_CONCURRENT" \
  --api_mode "$API_MODE" \


python -m src.evaluate.analyze_eval \
  --result-path "$RESULTS" \
  --metadata-path "$RECIPES" \
  --save-heatmaps \
  --output-dir "outputs/evaluation/captioning_evaluation"