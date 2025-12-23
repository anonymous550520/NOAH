RESULTS="path/to/qa_result.json"
OUTPUT="path/to/evaluate_result.json"


python -m src.evaluate.evaluate_qa \
  --results "$RESULTS" \
  --output "$OUTPUT"