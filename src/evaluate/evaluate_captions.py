import argparse
import json
import asyncio
import time
from src.evaluate.evaluators.hallucination import evaluate_hallucination_async
from src.evaluate.evaluators.omission import evaluate_omission_async
from src.evaluate.evaluators.prompt_loader import (
    load_hallucination_prompt,
    load_omission_prompt,
)
from src.utils.batch_api import enqueue_chat_completions_batch
from pathlib import Path

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def to_dict_by_id(data):
    return {r["id"]: r for r in data}

async def run_evaluation(mode, judge_model, recipes, results, output, max_concurrent, api_mode="sync"):
    # Preserve original file paths for metadata/logging
    recipes_path_str = str(recipes)
    results_path_str = str(results)
    # Single run timestamp anchored at evaluate invocation time
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    recipes = load_json(recipes_path_str)
    results = load_json(results_path_str)
    
    recipes_lookup = to_dict_by_id(recipes)
    results_lookup = to_dict_by_id(results)


    # Enqueue-only path: send batches and exit without parsing/saving
    if api_mode == "batch" and judge_model.lower().startswith("gpt"):
        out_path = Path(str(output))
        gp = out_path.parent.parent
        pp = out_path.parent
        if gp is not None and pp is not None and gp.name and pp.name:
            run_name = f"{gp.name}_{pp.name}"
        else:
            run_name = "run"

        def build_prompts(which: str):
            ids = sorted(recipes_lookup.keys() & results_lookup.keys())
            reqs = []
            for id in ids:
                recipe = recipes_lookup[id]
                result = results_lookup[id]
                if which == "hallucination":
                    prompt = load_hallucination_prompt(recipe, result)
                else:
                    prompt = load_omission_prompt(recipe, result)
                reqs.append((str(id), prompt))
            return reqs

        print(f"[enqueue] mode={mode}, api_mode={api_mode}")
        if mode in ("hallucination", "joint"):
            h_reqs = build_prompts("hallucination")
            if h_reqs:
                h_handles = enqueue_chat_completions_batch(
                    h_reqs,
                    mode="hallucination",
                    run_name=run_name,
                    recipes_path=str(Path(recipes_path_str)),
                    results_path=str(Path(results_path_str)),
                    ts=run_ts,
                    desired_output_path=str(Path(output)),
                )
                print(f"[enqueued] hallucination batch_id={h_handles.batch_id} dir={h_handles.root_dir} prefix={h_handles.prefix}")
            else:
                print("[enqueue] hallucination: no IDs to enqueue")

        if mode in ("omission", "joint"):
            o_reqs = build_prompts("omission")
            if o_reqs:
                o_handles = enqueue_chat_completions_batch(
                    o_reqs,
                    mode="omission",
                    run_name=run_name,
                    recipes_path=str(Path(recipes_path_str)),
                    results_path=str(Path(results_path_str)),
                    ts=run_ts,
                    desired_output_path=str(Path(output)),
                )
                print(f"[enqueued] omission batch_id={o_handles.batch_id} dir={o_handles.root_dir} prefix={o_handles.prefix}")
            else:
                print("[enqueue] omission: no IDs to enqueue")

        print("[done] Enqueued batches. Use batch worker to download and parse when completed.")
        return

    # Evaluate
    print(f"Starting evaluation: mode = {mode}, api_mode = {api_mode}")
    if mode == "hallucination":
        results = await evaluate_hallucination_async(judge_model, recipes_lookup, results_lookup, max_concurrent)
    elif mode == "omission":
        results = await evaluate_omission_async(judge_model, recipes_lookup, results_lookup, max_concurrent)
    elif mode == "joint":
        h_results = await evaluate_hallucination_async(judge_model, recipes_lookup, results_lookup, max_concurrent)
        o_results = await evaluate_omission_async(judge_model, recipes_lookup, results_lookup, max_concurrent)

        h_results = to_dict_by_id(h_results)
        o_results = to_dict_by_id(o_results)
        merged_ids = h_results.keys() & o_results.keys()

        results = []
        for id in merged_ids:
            result_item = {
                "id": id,
                "caption": results_lookup[id]["caption"],
                "ground_truth": recipes_lookup[id]["sentences"],
                "hallucination": {
                    "extracted_events": h_results[id]["hallucination_extracted_events"],
                    "extracted_events_count": len(h_results[id]["hallucination_extracted_events"]),
                    "hallucination_count": h_results[id]["hallucination_count"],
                    "reasoning": h_results[id]["hallucination_reasoning"],
                    "parse_error": h_results[id]["parse_error"],
                    "raw": h_results[id]["hallucination_raw"],
                },
                "omission": {
                    "ground_truth_events": o_results[id]["omission_ground_truth_events"],
                    "total_omission_count": o_results[id]["total_omission_count"],
                    "ground_truth_events_count": len(o_results[id]["omission_ground_truth_events"]),
                    "reasoning": o_results[id]["omission_reasoning"],
                    "parse_error": o_results[id]["parse_error"],
                    "raw": o_results[id]["omission_raw"],
                },
            }

            # Include inserted_omission_count
            result_item["omission"]["inserted_omission_count"] = o_results[id]["inserted_omission_count"]

            results.append(result_item)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Save
    print(f"Saving to {output}")
    save_json(results, output)
    print(f"Done. Total: {len(results)} evaluated.")

def parse_args():
    arg = argparse.ArgumentParser("Evaluate hallucination and omission errors")
    arg.add_argument("--mode", type=str, default="joint", choices=["hallucination", "omission", "joint"])
    arg.add_argument("--judge_model", type=str, default="gemini-2.0-flash", help="Model for judgment (default: gemini-2.0-flash)")
    arg.add_argument("--recipes", default="path/to/metadata.json")
    arg.add_argument("--results", default="outputs/captions/video_llama3_7b/captions_result.json")
    arg.add_argument("--output", default="outputs/captions/video_llama3_7b/evaluation_captions_result.json")
    arg.add_argument("--max_concurrent", type=int, default=20)
    arg.add_argument("--api_mode", type=str, default="sync", choices=["sync", "batch"], help="API call mode; default is sync (current behavior)")
    return arg.parse_args()

def main():
    args = parse_args()
    asyncio.run(run_evaluation(args.mode, args.judge_model, args.recipes, args.results, args.output, args.max_concurrent, args.api_mode))

if __name__ == "__main__":
    main()