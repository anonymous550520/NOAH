import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

"""Simple shard merging script.

Features:
	1. Load all JSON files matching pattern (task=captioning -> captioning_result_shard_*.json,
											task=qa         -> qa_result_shard_*.json)
	2. Concatenate lists as-is (no dedup / sorting / statistics)
	3. Save a single merged JSON file

Examples:
	python src/merge_shards.py --input-dir outputs/captions/video_llama3_7b/run_20250903_120000 --task captioning
	python src/merge_shards.py --input-dir outputs/qa/video_llama3_7b/run_20250903_120000 --task qa

With custom output name:
	python src/merge_shards.py --input-dir ... --task captioning --output merged.json
"""

CAPTIONING_PATTERN = "captioning_result_shard_*.json"
QA_PATTERN = "qa_result_shard_*.json"


def load_file(path: Path) -> List[Dict[str, Any]]:
	with open(path, "r") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Shard file must contain a list (file={path})")
	return data


def merge(args):
	input_dir = Path(args.input_dir)
	if not input_dir.is_dir():
		raise SystemExit(f"Input dir not found: {input_dir}")

	task_lower = args.task.lower()
	if task_lower in ("caption", "captioning"):
		pattern = CAPTIONING_PATTERN
		default_output_name = "captioning_result.json"
	elif task_lower in ("qa",):
		pattern = QA_PATTERN
		default_output_name = "qa_result.json"
	else:
		raise SystemExit("--task must be one of: captioning, qa")

	shard_files = sorted(input_dir.glob(pattern))
	if not shard_files:
		raise SystemExit(f"No shard files found (pattern={pattern}, dir={input_dir})")

	merged: List[Dict[str, Any]] = []
	for fp in shard_files:
		part = load_file(fp)
		merged.extend(part)

	output_path = Path(args.output) if args.output else input_dir / default_output_name
	with open(output_path, "w") as f:
		json.dump(merged, f, indent=2, ensure_ascii=False)

	print(f"Merged {len(shard_files)} files -> total records: {len(merged)}")
	print(f"Output: {output_path}")


def build_argparser():
	p = argparse.ArgumentParser(description="Simple shard result merger")
	p.add_argument("--input-dir", required=True, help="Directory containing shard result JSON files")
	p.add_argument("--task", required=True, help="captioning or qa")
	p.add_argument("--output", help="Output file path (default: *result.json inside the directory)")
	return p


def main():
	args = build_argparser().parse_args()
	merge(args)


if __name__ == "__main__":
	main()
