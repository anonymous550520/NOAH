"""
Utilities for using OpenAI Batch API without changing existing sync/async code paths.

This module prepares a JSONL input for /v1/chat/completions requests, uploads it
to the Files API with purpose "batch", creates a batch job, polls its status, and
retrieves & parses the results into a mapping of custom_id -> assistant content.

Important invariants for result parity:
- Uses the same endpoint (/v1/chat/completions) as existing code.
- Uses identical request body fields to current calls: model="gpt-4o",
  messages=[{"role": "user", "content": prompt}], temperature=0.0.
- Does not set extra parameters like max_tokens/top_p/etc.

Note: This module is not wired yet. It's added to enable a controlled, opt-in
transition in later steps. You can import and call run_chat_completions_batch()
from evaluation code once you switch to api_mode="batch".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

@dataclass
class BatchHandles:
    """Holds IDs and file paths for a created batch run."""

    input_jsonl_path: Path
    input_file_id: str
    batch_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    output_path: Optional[Path] = None
    error_path: Optional[Path] = None
    # Optional bookkeeping for enqueue-only flow
    root_dir: Optional[Path] = None
    prefix: Optional[str] = None
    meta_path: Optional[Path] = None
    batch_id_path: Optional[Path] = None


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_field(obj, name: str):
    """Safely get attribute or dict key from SDK objects or dicts."""
    try:
        return getattr(obj, name)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(name)
    return None



def _save_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_prefix(run_name: str, mode: str, model: str, ts: Optional[str] = None) -> str:
    use_ts = ts
    return f"{use_ts}-{run_name}-{mode}-{model}-full"


def build_chat_completions_jsonl(
    requests: Sequence[Tuple[str, str]],
    out_path: Path,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> Path:
    """
    Build a .jsonl file for /v1/chat/completions batch.

    requests: sequence of (custom_id, prompt) pairs.
    out_path: target file path to write jsonl.
    """
    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for custom_id, prompt in requests:
            line = {
                "custom_id": str(custom_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": str(prompt)},
                    ],
                    "temperature": float(temperature),
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out_path


def enqueue_chat_completions_batch(
    requests: Sequence[Tuple[str, str]],
    *,
    mode: str,
    run_name: str,
    recipes_path: str,
    results_path: str,
    ts: Optional[str] = None,
    desired_output_path: Optional[str] = None,
    work_root: Optional[Path] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    completion_window: str = "24h",
    extra_metadata: Optional[Dict[str, str]] = None,
) -> BatchHandles:
    """
    Enqueue-only helper: writes JSONL, uploads to Files API, and creates a batch.
    Does NOT poll or download results.

    Returns BatchHandles with paths and identifiers. output/error paths are None.
    """
    # Directory and prefix
    # Use RESULTS filename (without extension) as the grouping folder instead of model name
    results_stem = Path(results_path).stem if results_path else model
    root = (
        ((Path.cwd() / "outputs" / "batch_runs" / run_name / f"{results_stem}_{ts}" / mode)
        if work_root is None
        else Path(work_root))
    )
    _ensure_dir(root)
    # Keep file prefix unchanged (still includes model name) to avoid breaking filenames
    prefix = _build_prefix(run_name, mode, model, ts=ts)

    # 1) Build JSONL at prefixed path
    input_path = root / f"{prefix}-input.jsonl"
    build_chat_completions_jsonl(requests, input_path, model=model, temperature=temperature)

    # 2) Upload file and 3) Create batch
    input_file_id = upload_batch_input(input_path)
    meta = {
        "mode": mode,
        "model": model,
        "temperature": temperature,
        "run_name": run_name,
        "run_ts": ts,
        "recipes_path": recipes_path,
        "results_path": results_path,
        "desired_output_path": desired_output_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": "/v1/chat/completions",
        "api_flow": "enqueue-only",
    }
    if extra_metadata:
        meta.update(extra_metadata)

    batch_id = create_batch(input_file_id, completion_window=completion_window, metadata={k: str(v) for k, v in meta.items()})

    # Save meta and batch id locally
    meta_path = root / f"{prefix}-meta.json"
    _save_json(meta_path, {**meta, "input_file_id": input_file_id, "batch_id": batch_id})
    batch_id_path = root / f"{prefix}-batch.txt"
    batch_id_path.write_text(batch_id + "\n", encoding="utf-8")

    return BatchHandles(
        input_jsonl_path=input_path,
        input_file_id=input_file_id,
        batch_id=batch_id,
        output_file_id=None,
        error_file_id=None,
        output_path=None,
        error_path=None,
        root_dir=root,
        prefix=prefix,
        meta_path=meta_path,
        batch_id_path=batch_id_path,
    )


def upload_batch_input(file_path: Path) -> str:
    """Upload the JSONL file to Files API with purpose="batch" and return file id."""
    client = _get_client()
    with file_path.open("rb") as fp:
        file_obj = client.files.create(file=fp, purpose="batch")
    return file_obj.id


def create_batch(
    input_file_id: str,
    *,
    completion_window: str = "24h",
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Create a batch for chat completions and return the batch id."""
    client = _get_client()
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata=metadata or None,
    )
    
    return batch.id


def retrieve_batch(batch_id: str):
    client = _get_client()
    return client.batches.retrieve(batch_id)


def poll_batch_until_done(
    batch_id: str,
    *,
    poll_interval_sec: float = 5.0,
    timeout_sec: Optional[float] = None,
) -> dict:
    """
    Poll a batch until it reaches a terminal state and return the final batch object.

    Terminal states: completed, failed, expired, cancelled.
    """
    start = time.time()
    client = _get_client()

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = _get_field(batch, "status")
        if status in terminal:
            return batch
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            return batch
        time.sleep(poll_interval_sec)


def _download_file_content(file_id: str) -> str:
    client = _get_client()
    file_response = client.files.content(file_id)
    # The SDK returns a response-like object whose .text holds content
    return file_response.text  # type: ignore[attr-defined]


def download_results_files(
    batch_obj: dict,
    *,
    out_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download output and error files (if any) to out_dir. Returns their paths.
    """
    _ensure_dir(out_dir)
    output_file_id = _get_field(batch_obj, "output_file_id")
    error_file_id = _get_field(batch_obj, "error_file_id")

    output_path = None
    error_path = None

    if output_file_id:
        text = _download_file_content(output_file_id)
        output_path = out_dir / "batch_output.jsonl"
        output_path.write_text(text, encoding="utf-8")

    if error_file_id:
        text = _download_file_content(error_file_id)
        error_path = out_dir / "batch_errors.jsonl"
        error_path.write_text(text, encoding="utf-8")

    return output_path, error_path


def parse_batch_output_jsonl(path: Path) -> Dict[str, str]:
    """
    Parse the batch output .jsonl and build custom_id -> assistant content map.
    """
    result: Dict[str, str] = {}
    if not path or not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            resp = obj.get("response") or {}
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if custom_id is not None:
                    result[str(custom_id)] = content
    return result


def parse_batch_errors_jsonl(path: Optional[Path]) -> List[str]:
    """Return list of custom_ids that failed/expired from the error file."""
    failed: List[str] = []
    if not path or not path.exists():
        return failed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            if custom_id is not None:
                failed.append(str(custom_id))
    return failed


def run_chat_completions_batch(
    requests: Sequence[Tuple[str, str]],
    *,
    work_dir: Optional[Path] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    completion_window: str = "24h",
    metadata: Optional[Dict[str, str]] = None,
    poll_interval_sec: float = 5.0,
    timeout_sec: Optional[float] = None,
):
    """
    High-level helper: run a full batch job and return results.

    Returns:
      - results_map: {custom_id: assistant_content}
      - handles: IDs and file paths for bookkeeping
      - failed_ids: custom_ids that appear in error file (can be retried sync)
    """
    # Prepare directories
    ts = time.strftime("%Y%m%d-%H%M%S")
    root = Path.cwd() / "experiments" / "batch_runs" / ts if work_dir is None else Path(work_dir)
    _ensure_dir(root)

    # 1) Build JSONL
    input_path = root / "batch_input.jsonl"
    build_chat_completions_jsonl(requests, input_path, model=model, temperature=temperature)

    # 2) Upload
    input_file_id = upload_batch_input(input_path)

    # 3) Create batch
    batch_id = create_batch(input_file_id, completion_window=completion_window, metadata=metadata)
    # 4) Poll until done
    final_batch = poll_batch_until_done(batch_id, poll_interval_sec=poll_interval_sec, timeout_sec=timeout_sec)

    # 5) Download results
    output_path, error_path = download_results_files(final_batch, out_dir=root)

    # 6) Parse
    results_map = parse_batch_output_jsonl(output_path) if output_path else {}
    failed_ids = parse_batch_errors_jsonl(error_path)

    handles = BatchHandles(
        input_jsonl_path=input_path,
        input_file_id=input_file_id,
        batch_id=batch_id,
        output_file_id=_get_field(final_batch, "output_file_id"),
        error_file_id=_get_field(final_batch, "error_file_id"),
        output_path=output_path,
        error_path=error_path,
    )

    return results_map, handles, failed_ids
