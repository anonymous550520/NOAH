import asyncio
from tqdm.asyncio import tqdm
from src.evaluate.evaluators.prompt_loader import load_hallucination_prompt
from src.utils.parse import parse_hallucination_text
from src.utils.api import call_judge_model
from src.utils.batch_api import run_chat_completions_batch

async def evaluate_single_hallucination(id, recipe, result, semaphore, judge_model=None):
    async with semaphore:
        try:
            # Load and format the prompt
            prompt = load_hallucination_prompt(recipe, result)
            
            # Call the judge model
            raw_output = await call_judge_model(prompt, model=judge_model)

            # Parse the structured response
            parsed_output = parse_hallucination_text(raw_output)

            return {
                "id": id,
                "caption": result.get("caption", ""),
                "hallucination_count": parsed_output.get("hallucination_count", 0),
                "hallucination_reasoning": parsed_output.get("reasoning", ""),
                "hallucination_extracted_events": parsed_output.get("extracted_events", []),
                "hallucination_raw": raw_output,
                "parse_error": parsed_output.get("parse_error")
            }
        except Exception as e:
            print(f"[Error] ID={id} | {e}")
            return {
                "id": id,
                "caption": result.get("caption", ""),
                "hallucination_count": 0,
                "hallucination_reasoning": "",
                "hallucination_extracted_events": [],
                "hallucination_raw": None,
                "error": str(e),
                "parse_error": None
            }

async def evaluate_hallucination_async(
    judge_model,
    recipe_lookup,
    result_lookup,
    max_concurrent,
):
    """
    Evaluate hallucination errors.

    api_mode:
      - "sync": existing behavior (per-sample async calls with semaphore)
      - "batch": use OpenAI Batch API for bulk calls, then parse; fallback to sync for failures
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    ids = sorted(recipe_lookup.keys() & result_lookup.keys())


    tasks = []
    for id in ids:
        recipe = recipe_lookup[id]
        result = result_lookup[id]
        tasks.append(evaluate_single_hallucination(id, recipe, result, semaphore, judge_model))

    results = []
    for coro in tqdm.as_completed(tasks, desc="Evaluating Hallucination", total=len(tasks)):
        result = await coro
        results.append(result)
    return results