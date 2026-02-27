#!/usr/bin/env python3
"""
Run src/main.py pipeline for a single sentence and write an execution log + summary.

This is intentionally *not* a pytest test:
- the pipeline is heavy (GiNZA, BERT, GPU, LLM) and pytest timeouts/output capture are painful.
- for reproducibility we write a self-contained run directory with logs + summary.json.

Expected environment:
- mask BERT (tohoku-nlp/bert-base-japanese-v3) cached in HF_HOME
- dep-bert local model exists (DEP_BERT_MODEL_PATH or default path in CKYAnalyzer)
- llm-jp is reachable via existing LLMJP_BASE_URL(S) / LLMJP_MODEL env vars
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path("/workspace")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_patterns_from_pkl(out_dir: Path) -> Tuple[Optional[Path], Optional[Path], int]:
    pkl_path = REPO_ROOT / "data/patterns/patterns_ast.pkl.gz"
    if not pkl_path.exists():
        return None, None, 0

    sys.path.append(str(REPO_ROOT / "src"))
    import pattern.pattern_nodes as pn  # noqa: F401

    # legacy unpickling compatibility
    sys.modules["pattern_nodes"] = pn

    with gzip.open(pkl_path, "rb") as f:
        patterns = pickle.load(f)

    index_rows = []
    jsonl_rows = []
    for i, entry in enumerate(patterns):
        pattern = entry.get("pattern")
        if not pattern:
            continue
        pid = f"PKL_{i}"
        index_rows.append({"pattern_id": pid, "status": "active"})
        jsonl_rows.append({"pattern_id": pid, "status": "active", "pattern": pattern})

    if not index_rows:
        return None, None, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "patterns.index.json"
    jsonl_path = out_dir / "patterns.jsonl"
    write_json(index_path, {"count": len(index_rows), "patterns": index_rows})
    write_jsonl(jsonl_path, jsonl_rows)
    return index_path, jsonl_path, len(index_rows)


def find_record_by_id(data_dir: Path, target_id: str) -> Tuple[Optional[Path], Optional[dict]]:
    for path in sorted(data_dir.glob("*.jsonl")):
        for obj in iter_jsonl(path):
            if str(obj.get("id", "")) == target_id:
                return path, obj
    return None, None


def pick_default_record(data_dir: Path) -> Tuple[Optional[Path], Optional[dict]]:
    # Prefer a stable movie example if present.
    movie_path = data_dir / "ont_1_movie_ground_truth_target.jsonl"
    if movie_path.exists():
        preferred_ids = ["ont_1_movie_test_13"]
        for pid in preferred_ids:
            p, rec = find_record_by_id(data_dir, pid)
            if rec is not None:
                return p, rec
        for obj in iter_jsonl(movie_path):
            sent = obj.get("sent_ja") or ""
            if sent:
                return movie_path, obj

    # Fallback: first record in first jsonl.
    for path in sorted(data_dir.glob("*.jsonl")):
        for obj in iter_jsonl(path):
            sent = obj.get("sent_ja") or obj.get("sent") or ""
            if sent:
                return path, obj
    return None, None


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def ensure_hf_snapshot(repo_id: str, hf_home: Path) -> str:
    """
    Resolve a cached snapshot path for repo_id.
    This must not download: we set OFFLINE flags, and fail fast if missing.
    """
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=str(hf_home), local_files_only=True)


def tee_run(cmd: list[str], *, cwd: Path, env: dict, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[cmd] {' '.join(cmd)}\n")
        f.write(f"[cwd] {cwd}\n")
        f.write("[env]\n")
        for k in sorted(env.keys()):
            if k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "LLMJP_API_KEY"):
                continue
            f.write(f"{k}={env[k]}\n")
        f.write("\n[output]\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            # Stream to console and to file (simple & reliable).
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
        dt = time.time() - t0
        f.write(f"\n[exit] rc={rc} elapsed_sec={dt:.3f}\n")
        f.flush()
        if rc != 0:
            raise SystemExit(f"main failed with rc={rc}. See log: {log_path}")
        return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="target_id", default="", help="target_data 内の id（未指定なら適当な1件）")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/T2KGB_JA/target_data"))
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "tmp_single_sentence_run"))
    ap.add_argument("--keep", action="store_true", help="run-dir を消さずに上書きしない")
    ap.add_argument("--mask-model", default=os.getenv("MASK_BERT_MODEL_NAME", "tohoku-nlp/bert-base-japanese-v3"))
    ap.add_argument("--hf-home", default=os.getenv("HF_HOME", str(REPO_ROOT / ".cache/huggingface")))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    hf_home = Path(args.hf_home)

    if (not args.keep) and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pick one record
    if args.target_id:
        src_path, record = find_record_by_id(data_dir, args.target_id)
    else:
        src_path, record = pick_default_record(data_dir)
    if src_path is None or record is None:
        raise SystemExit("対象データが見つかりません")

    sentence = record.get("sent_ja") or record.get("sent") or ""
    if not sentence:
        raise SystemExit("sent_ja/sent が空です")

    ontology_id = str(record.get("ontology_id") or "")

    # Prepare patterns + input
    patterns_dir = run_dir / "patterns"
    pattern_index, pattern_jsonl, pattern_count = build_patterns_from_pkl(patterns_dir)
    if pattern_index is None or pattern_jsonl is None:
        raise SystemExit("patterns_ast.pkl.gz からパターン生成できません")

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / src_path.name
    write_jsonl(
        input_path,
        [
            {
                "id": record.get("id", ""),
                "sent_ja": sentence,
                "ontology_id": ontology_id,
            }
        ],
    )

    # Ensure mask model snapshot exists locally and force transformers to use it.
    t_cache0 = time.time()
    snapshot_path = ensure_hf_snapshot(args.mask_model, hf_home)
    t_cache = time.time() - t_cache0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PATTERN_INDEX_JSON"] = str(pattern_index)
    env["PATTERN_JSONL"] = str(pattern_jsonl)
    env["INPUT_JSONL_DIR"] = str(input_dir)
    env["RESULTS_ROOT"] = str(run_dir / "results")
    env["EXPORT_AST_REPR"] = "1"

    # Force mask BERT to load from local snapshot (no hub access).
    env["MASK_BERT_MODEL_NAME"] = args.mask_model
    env["MASK_BERT_MODEL_PATH"] = snapshot_path
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # Lightweight stats/progress for model init if modules support it.
    env.setdefault("SHOW_MODEL_PROGRESS", "1")
    # Prevent CKY candidate explosion (tree candidate cross-products) from stalling the GPU worker.
    env.setdefault("CKY_MAX_CHILD_CANDIDATES", "1")
    env.setdefault("CKY_MAX_CELL_CANDIDATES_TOTAL", "64")

    # Reduce LLM hangs: keep existing base_url/model from environment; just cap retries/timeouts if not set.
    env.setdefault("LLMJP_TIMEOUT_SEC", "30")
    env.setdefault("LLMJP_MAX_RETRIES", "1")
    env.setdefault("LLMJP_BACKOFF_SEC", "0.5")

    # Run pipeline
    log_path = run_dir / "run.log"
    t_total = tee_run(
        ["python", "-c", "import src.main as m; m.main()"],
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
    )

    # Summarize outputs
    out_dir = (run_dir / "results") / "input" / src_path.stem
    logs_dir = out_dir / "logs"
    summary: Dict[str, Any] = {
        "input": {
            "source_jsonl": str(src_path),
            "id": record.get("id", ""),
            "ontology_id": ontology_id,
            "sentence": sentence,
        },
        "cache": {
            "hf_home": str(hf_home),
            "mask_model": args.mask_model,
            "mask_snapshot": snapshot_path,
            "ensure_snapshot_sec": t_cache,
        },
        "patterns": {
            "count": pattern_count,
            "pattern_index": str(pattern_index),
            "pattern_jsonl": str(pattern_jsonl),
        },
        "run": {
            "run_dir": str(run_dir),
            "results_dir": str(out_dir),
            "run_log": str(log_path),
            "elapsed_sec": t_total,
        },
        "outputs": {},
        "logs": {},
        "warnings": [],
    }

    candidate_csv = out_dir / f"{src_path.stem}_triples_candidate.csv"
    verified_csv = out_dir / f"{src_path.stem}_triples_verified.csv"
    vis_csv = out_dir / f"{src_path.stem}_ast_visualization.csv"
    prompt_log = out_dir / f"{src_path.stem}_prompt_log.jsonl"
    summary["outputs"] = {
        "candidate_csv": str(candidate_csv),
        "verified_csv": str(verified_csv),
        "vis_csv": str(vis_csv),
        "prompt_log": str(prompt_log),
        "candidate_rows": count_csv_rows(candidate_csv),
        "verified_rows": count_csv_rows(verified_csv),
        "vis_rows": count_csv_rows(vis_csv),
        "prompt_rows": sum(1 for _ in prompt_log.open("r", encoding="utf-8")) if prompt_log.exists() else 0,
    }

    gpu_timing_csv = logs_dir / f"{src_path.stem}_gpu_timing.csv"
    cpu_timing_csv = logs_dir / f"{src_path.stem}_cpu_timing.csv"
    inflight_csv = logs_dir / f"{src_path.stem}_inflight.csv"
    summary["logs"] = {
        "gpu_timing_csv": str(gpu_timing_csv),
        "cpu_timing_csv": str(cpu_timing_csv),
        "inflight_csv": str(inflight_csv),
        "gpu_timing_rows": len(read_csv_rows(gpu_timing_csv)),
        "cpu_timing_rows": len(read_csv_rows(cpu_timing_csv)),
        "inflight_rows": len(inflight_csv.read_text(encoding="utf-8").splitlines()) if inflight_csv.exists() else 0,
    }

    if summary["outputs"]["prompt_rows"] <= 0:
        summary["warnings"].append("prompt_log is empty (LLM might not have been called).")
    if summary["logs"]["gpu_timing_rows"] <= 0:
        summary["warnings"].append("gpu_timing has no rows (GPU worker may not have completed).")
    if summary["logs"]["cpu_timing_rows"] <= 0:
        summary["warnings"].append("cpu_timing has no rows (CPU stage may not have completed).")

    write_json(run_dir / "summary.json", summary)
    print(f"\n[done] summary: {run_dir / 'summary.json'}")
    if summary["warnings"]:
        print("[warnings]")
        for w in summary["warnings"]:
            print(f"- {w}")


if __name__ == "__main__":
    main()
