import json
import argparse
from pathlib import Path
from tqdm import tqdm
from model_loader import get_model_module

BINARY_QA_INSTRUCTION="Answer with 'yes' or 'no'."

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def save_results(results, save_dir, shard_idx):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"qa_result_shard_{shard_idx}.json", "w") as f:
        json.dump(results, f, indent=2)

def process_qa(args):
    if args.existence_qa_path is None:
        existence_qa = []
    else:
        existence_qa = load_json(args.existence_qa_path)
    if args.temporal_qa_path is None:
        temporal_qa = []
    else:
        temporal_qa = load_json(args.temporal_qa_path)
    if args.narrative_qa_path is None:
        narrative_qa = []
    else:
        narrative_qa = load_json(args.narrative_qa_path)

    qa = existence_qa + temporal_qa + narrative_qa
    if args.retry_result_path is not None:
        retry_result = load_json(args.retry_result_path)
        for result in retry_result:
            for item in qa:
                if item['id'] == result['id'] and item['type'] == result['type']:
                    item['result'] = result
    model_module = get_model_module(args.model_name)
    model_info = model_module.load_model(args.model_name)

    #qa sort by have result last
    
    shard_qa = qa[args.shard_idx :: args.num_shards]
    shard_qa.sort(key=lambda x: 'result' in x)

    results=[]

    for i, item in enumerate(tqdm(shard_qa, desc=f"Generating qa answers with {args.model_name}")):
        if 'result' in item and item['result']['status'] == 'success':
            results.append(item['result'])
            print(f"[Skip] type={item['type']}, id={item['id']}  already succeeded in previous run")
        else:
            video_path = f"{args.dataset_dir}/{item['video_id']}.mp4"

            question = f"{item['question']} {BINARY_QA_INSTRUCTION}"

            try:
                predict = model_module.generate(model_info, video_path, question, max_new_tokens=128)
                
                results.append({
                    "id": item['video_id'],
                    "type": item['type'],
                    "question": question,
                    "answer": item['answer'],
                    "predict": predict,
                    "status": "success"
                })

            except Exception as e:
                results.append({
                    "id": item['video_id'],
                    "type": item['type'],
                    "question": question,
                    "answer": item['answer'],
                    "predict": None,
                    "status": "fail",
                    "error": str(e)
                })
                print(f"Error: {e}")

        if (i + 1) % 100 == 0:
            save_results(results, args.save_dir, args.shard_idx)
    
    save_results(results, args.save_dir, args.shard_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--existence-qa-path", type=str, default=None)
    parser.add_argument("--temporal-qa-path", type=str, default=None)
    parser.add_argument("--narrative-qa-path", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx",  type=int, default=0)
    parser.add_argument("--retry-result-path", type=str, default=None)

    args = parser.parse_args()

    process_qa(args)
        
if __name__ == "__main__":
    main()