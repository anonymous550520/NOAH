RESULTS="path/to/qa_result.json"
OUTPUT="path/to/evaluate_result.json"


python -m src.evaluate.evaluate_qa_pairwise \
  --results "$RESULTS" \
  --output "$OUTPUT"