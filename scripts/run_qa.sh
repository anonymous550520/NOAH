#!/bin/bash
set -euo pipefail
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
# usage:
#   ./scripts/run_captioning.sh MODEL_NAME [WORKERS_PER_GPU]
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_captioning.sh MODEL_NAME [WORKERS_PER_GPU]
# example:
#   ./scripts/run_captioning.sh video_llama3_7b
#   ./scripts/run_captioning.sh video_llama3_7b 2

MODEL_NAME=${1:?MODEL_NAME required}
WORKERS_PER_GPU=${2:-1}

DATASET_DIR="path/to/dataset"
NARRATIVE_QA_PATH="path/to/narrative_qa.json"
EXISTENCE_QA_PATH="path/to/existence_qa.json"
TEMPORAL_QA_PATH="path/to/temporal_qa.json"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="outputs/qa/${MODEL_NAME}/run_${TIMESTAMP}"
mkdir -p "$SAVE_DIR"

# --- logging ---
LOG_FILE="$SAVE_DIR/run.log"
touch "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1
echo "INFO: Logging only to $LOG_FILE (no console echo)"
trap 'echo "INFO: Script finished at $(date)"' EXIT
# --- logging ---

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: no CUDA_VISIBLE_DEVICES and nvidia-smi not found." >&2
        exit 1
    fi
    NG=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [[ "$NG" -lt 1 ]]; then
        echo "ERROR: no GPUs detected." >&2
        exit 1
    fi
    mapfile -t GPUS_TO_USE < <(seq 0 $((NG - 1)))
else
    IFS=',' read -r -a GPUS_TO_USE <<< "$CUDA_VISIBLE_DEVICES"
    NG=${#GPUS_TO_USE[@]}
fi

NUM_SHARDS=$(( NG * WORKERS_PER_GPU ))


echo "Using GPUs: [${GPUS_TO_USE[*]}], workers_per_gpu=${WORKERS_PER_GPU} -> num_shards=${NUM_SHARDS}"

SHARD_IDX=0
for GPU_ID in "${GPUS_TO_USE[@]}"; do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    echo "  -> Launching shard ${SHARD_IDX} (gpu=${GPU_ID}, worker=${w})"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python src/run_qa.py \
      --model-name "$MODEL_NAME" \
      --dataset-dir "$DATASET_DIR" \
      --existence-qa-path "$EXISTENCE_QA_PATH" \
      --temporal-qa-path "$TEMPORAL_QA_PATH" \
      --narrative-qa-path "$NARRATIVE_QA_PATH" \
      --num-shards "$NUM_SHARDS" \
      --shard-idx "$SHARD_IDX" \
      --save-dir "$SAVE_DIR" &
    SHARD_IDX=$((SHARD_IDX + 1))
  done
done

wait

python src/merge_shards.py \
  --input-dir "$SAVE_DIR" \
  --task qa

echo "INFO: All shards finished. Results at: $SAVE_DIR"