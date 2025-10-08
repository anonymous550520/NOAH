import asyncio
from tqdm.asyncio import tqdm
from src.evaluate.evaluators.prompt_loader import load_omission_prompt
from src.utils.parse import parse_omission_text
from src.utils.api import call_judge_model
from src.utils.batch_api import run_chat_completions_batch

async def evaluate_single_omission(id, recipe, result, semaphore, judge_model=None):
    async with semaphore:
        try:
            # Load and format the prompt
            prompt = load_omission_prompt(recipe, result)
            
            # Call the judge model
            raw_output = await call_judge_model(prompt, model=judge_model)
            
            # Parse the structured response
            parsed_output = parse_omission_text(raw_output)

            result_data = {
                "id": id,
                "omission": parsed_output.get("total_omission_count", 0) > 0,
                "total_omission_count": parsed_output.get("total_omission_count", 0),
                "omission_reasoning": parsed_output.get("reasoning", ""),
                "omission_ground_truth_events": parsed_output.get("ground_truth_events", []),
                "omission_raw": raw_output,
                "parse_error": parsed_output.get("parse_error")
            }
            
            # Include inserted_omission_count
            result_data["inserted_omission_count"] = parsed_output.get("inserted_omission_count", 0)
            
            return result_data
        except Exception as e:
            print(f"[Error] ID={id} | {e}")
            error_result = {
                "id": id,
                "omission": False,
                "total_omission_count": 0,
                "omission_reasoning": "",
                "omission_ground_truth_events": [],
                "omission_raw": None,
                "error": str(e),
                "parse_error": None
            }
            
            # Include inserted_omission_count
            error_result["inserted_omission_count"] = 0
                
            return error_result

async def evaluate_omission_async(
    judge_model,
    recipe_lookup,
    result_lookup,
    max_concurrent,
):
    """
    Evaluate omission errors.

    api_mode:
      - "sync": existing behavior (per-sample async calls)
      - "batch": use OpenAI Batch API; fallback to sync for failures
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    ids = sorted(recipe_lookup.keys() & result_lookup.keys())

    tasks = []
    for id in ids:
        recipe = recipe_lookup[id]
        result = result_lookup[id]
        tasks.append(evaluate_single_omission(id, recipe, result, semaphore, judge_model))

    results = []
    for coro in tqdm.as_completed(tasks, desc="Evaluating Omission", total=len(tasks)):
        result = await coro
        results.append(result)
    return results