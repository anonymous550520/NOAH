import json
import argparse
from pathlib import Path
from tqdm import tqdm
from model_loader import get_model_module
import time

DEFAULT_USER_QUERY = "What happens in this video? Please describe it in detail."

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def save_results(results, save_dir, shard_idx):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"captioning_result_shard_{shard_idx}.json", "w") as f:
        json.dump(results, f, indent=2)

def process_captioning(args):
    metadata = load_json(args.metadata_path)
    # Load retry results if provided
    if args.retry_result_path is not None:
        retry_result = load_json(args.retry_result_path)
        for result in retry_result:
            for item in metadata:
                if item['id'] == result['id']:
                    item['result'] = result
    model_module = get_model_module(args.model_name)
    model_info = model_module.load_model(args.model_name)
    # Add retry results for already successful IDs

    shard_metadata = metadata[args.shard_idx :: args.num_shards]

    results=[]

    for i, item in enumerate(tqdm(shard_metadata, desc=f"Generating captions with {args.model_name}")):
        # Skip items that already succeeded in previous runs
        if 'result' in item and item['result']['status'] == 'success':
            results.append(item['result'])
            print(f"[Skip] id={item['id']} already succeeded in previous run")
        elif 'result' in item and item['result']['status'] == 'fail' and item['result']['error'].startswith("Gemini blocked generation"):
            results.append(item['result'])
            print(f"[Skip] id={item['id']} blocked by Gemini in previous run")
        else:
            video_path = f"{args.dataset_dir}/{item['id']}.mp4"

            caption = None
            status = "fail"
            error_msg = None

            # max 1 attempts
            max_attempts = 1
            for attempt in range(1, max_attempts + 1):
                try:
                    caption = model_module.generate(model_info, video_path, DEFAULT_USER_QUERY)
                    status = "success"
                    if attempt > 1:
                        print(f"[Success try] id={item['id']} attempt={attempt}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"[Error] id={item['id']} attempt={attempt} error={error_msg}")
                    if attempt < max_attempts:
                        time.sleep(1 * attempt)
                    else:
                        print(f"[Fail] id={item['id']}")

            results.append({
                "id": item['id'],
                "caption": caption,
                "status": status,
                **({"error": error_msg} if status == "fail" else {})
            })

        if (i + 1) % 100 == 0:
            save_results(results, args.save_dir, args.shard_idx)
    
    save_results(results, args.save_dir, args.shard_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--metadata-path", type=str, default="path/to/metadata.json")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx",  type=int, default=0)
    parser.add_argument("--retry-result-path", type=str, default=None)

    args = parser.parse_args()

    process_captioning(args)
        
if __name__ == "__main__":
    main()