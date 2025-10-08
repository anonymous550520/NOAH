#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

from src.utils.batch_api import retrieve_batch, download_results_files, parse_batch_output_jsonl, parse_batch_errors_jsonl
from src.utils.parse import parse_hallucination_text, parse_omission_text


def _to_dict_by_id(items):
    out = {}
    for r in items or []:
        if isinstance(r, dict) and "id" in r:
            out[str(r["id"])] = r
    return out


def try_merge_joint(meta_path: Path, parsed_path: Path) -> None:
    """If the counterpart mode's parsed file exists, merge to a joint file.

    Joint file is written under model directory with a timestamped name.
    """
    try:
        meta = load_json(meta_path)
    except Exception:
        return

    mode = meta.get("mode")
    run_name = meta.get("run_name", "run")
    model = meta.get("model", "model")
    run_ts = meta.get("run_ts")  # stable timestamp passed from enqueue
    recipes_path = meta.get("recipes_path")
    results_path = meta.get("results_path")

    other_mode = "omission" if mode == "hallucination" else ("hallucination" if mode == "omission" else None)
    if not other_mode or not recipes_path or not results_path:
        return

    # Locate counterpart meta/parsed in sibling mode dir
    model_dir = meta_path.parent.parent  # .../{model}/
    other_dir = model_dir / other_mode
    if not other_dir.exists():
        return

    # Pick the latest counterpart that matches same dataset paths
    candidates = []
    for mp in other_dir.glob("*-meta.json"):
        try:
            m = load_json(mp)
        except Exception:
            continue
        if m.get("recipes_path") != recipes_path or m.get("results_path") != results_path:
            continue
        pp = mp.parent / (mp.name[:-10] + "-parsed.json")
        if pp.exists():
            candidates.append((pp.stat().st_mtime, mp, pp))

    if not candidates:
        return

    candidates.sort(reverse=True)
    _mtime, other_meta_path, other_parsed = candidates[0]

    # Load all inputs
    try:
        h_parsed = load_json(parsed_path if mode == "hallucination" else other_parsed)
        o_parsed = load_json(parsed_path if mode == "omission" else other_parsed)
        recipes = load_json(Path(recipes_path))
        results = load_json(Path(results_path))
        other_meta_obj = load_json(other_meta_path)
    except Exception as e:
        print(f"[merge] load error: {e}")
        return

    # Build dicts
    H = _to_dict_by_id(h_parsed)
    O = _to_dict_by_id(o_parsed)
    RCP = _to_dict_by_id(recipes)
    RES = _to_dict_by_id(results)

    merged_ids = sorted(H.keys() & O.keys() & RCP.keys() & RES.keys())
    out = []
    for _id in merged_ids:
        hitem = H[_id]
        oitem = O[_id]
        cap = RES[_id].get("caption", "")
        gt_sentences = RCP[_id].get("sentences", [])

        hallu_extracted = hitem.get("hallucination_extracted_events", [])
        omission_gt_events = oitem.get("omission_ground_truth_events", [])

        item = {
            "id": _id,
            "caption": cap,
            "ground_truth": gt_sentences,
            "hallucination": {
                "extracted_events": hallu_extracted,
                "extracted_events_count": len(hallu_extracted),
                "hallucination_count": hitem.get("hallucination_count", 0),
                "reasoning": hitem.get("hallucination_reasoning", ""),
                "parse_error": hitem.get("parse_error"),
                "raw": hitem.get("hallucination_raw"),
            },
            "omission": {
                "ground_truth_events": omission_gt_events,
                "total_omission_count": oitem.get("total_omission_count", 0),
                "ground_truth_events_count": len(omission_gt_events),
                "reasoning": oitem.get("omission_reasoning", ""),
                "parse_error": oitem.get("parse_error"),
                "raw": oitem.get("omission_raw"),
            },
        }
        if "inserted_omission_count" in oitem:
            item["omission"]["inserted_omission_count"] = oitem.get("inserted_omission_count", 0)

        out.append(item)

    # Decide output path: prefer desired_output_path from meta, then counterpart's, else default under model dir
    desired_path = meta.get("desired_output_path") or (other_meta_obj.get("desired_output_path") if isinstance(other_meta_obj, dict) else None)
    if desired_path:
        joint_path = Path(desired_path)
    else:
        ts = run_ts
        joint_path = model_dir / f"{ts}-{run_name}-joint-{model}-full.json"
    # Ensure parent directory exists for custom paths
    try:
        joint_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    save_json(out, joint_path)
    print(f"[merge] joint saved: {joint_path} (items={len(out)})")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def find_meta_files(root: Path):
    return list(root.glob("**/*-meta.json"))


def already_parsed(meta_path: Path) -> bool:
    prefix = meta_path.name[:-10]  # strip -meta.json
    parsed = meta_path.parent / f"{prefix}-parsed.json"
    return parsed.exists()


def process_one(meta_path: Path) -> Tuple[bool, str]:
    meta = load_json(meta_path)
    batch_id = meta.get("batch_id")
    mode = meta.get("mode")

    if not batch_id or not mode:
        return False, f"[skip] invalid meta: {meta_path}"

    # Check batch status
    batch = retrieve_batch(batch_id)
    status = getattr(batch, "status", None) or (batch.get("status") if isinstance(batch, dict) else None)
    if status not in {"completed", "failed", "expired", "cancelled"}:
        return False, f"[wait] batch {batch_id} status={status}"

    # Download outputs
    output_path, error_path = download_results_files(batch, out_dir=meta_path.parent)
    prefix = meta_path.name[:-10]
    parsed_path = meta_path.parent / f"{prefix}-parsed.json"

    # Build id -> raw text map
    results_map = parse_batch_output_jsonl(output_path) if output_path else {}
    failed_ids = set(parse_batch_errors_jsonl(error_path))

    # Parse to schema
    parsed_results = []
    if mode == "hallucination":
        for id, raw in results_map.items():
            po = parse_hallucination_text(raw)
            parsed_results.append({
                "id": id,
                "caption": "",
                "hallucination_count": po.get("hallucination_count", 0),
                "hallucination_reasoning": po.get("reasoning", ""),
                "hallucination_extracted_events": po.get("extracted_events", []),
                "hallucination_raw": raw,
                "parse_error": po.get("parse_error"),
            })
    elif mode == "omission":
        for id, raw in results_map.items():
            po = parse_omission_text(raw)
            item = {
                "id": id,
                "omission": po.get("total_omission_count", 0) > 0,
                "total_omission_count": po.get("total_omission_count", 0),
                "omission_reasoning": po.get("reasoning", ""),
                "omission_ground_truth_events": po.get("ground_truth_events", []),
                "omission_raw": raw,
                "parse_error": po.get("parse_error"),
            }
            item["inserted_omission_count"] = po.get("inserted_omission_count", 0)
            parsed_results.append(item)
    else:
        return False, f"[skip] unknown mode in meta: {mode}"

    save_json(parsed_results, parsed_path)
    try_merge_joint(meta_path, parsed_path)
    return True, f"[ok] parsed saved: {parsed_path} (failed_ids={len(failed_ids)})"


def main():
    ap = argparse.ArgumentParser("Batch worker: download completed batches and parse")
    ap.add_argument("--root", default="experiments/batch_runs", help="Root directory containing batch runs")
    ap.add_argument("--batch-id", default=None, help="Process only this batch id")
    ap.add_argument("--once", action="store_true", help="Run once and exit (no watch)")
    ap.add_argument("--interval", type=int, default=30, help="Watch polling interval seconds")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[error] root not found: {root}")
        return

    def eligible(mp: Path) -> bool:
        if args.batch_id:
            try:
                meta = load_json(mp)
            except Exception:
                return False
            return meta.get("batch_id") == args.batch_id
        return True

    while True:
        metas = [p for p in find_meta_files(root) if eligible(p) and not already_parsed(p)]
        if not metas:
            if args.once:
                print("[done] no metas to process")
                return
            time.sleep(args.interval)
            continue

        for mp in metas:
            ok, msg = process_one(mp)
            print(msg)

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
