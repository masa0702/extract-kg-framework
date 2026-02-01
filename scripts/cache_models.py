#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from pathlib import Path

from tqdm.auto import tqdm


def snapshot_with_progress(repo_id: str, cache_dir: Path) -> str:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    print(f"[cache] start: {repo_id}")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_files_only=False,
        resume_download=True,
        tqdm_class=tqdm,
    )
    dt = time.time() - t0
    print(f"[cache] done: {repo_id} ({dt:.1f}s) -> {path}")
    return path


def clean_unused(cache_dir: Path, keep_models: set[str]) -> None:
    hub_dir = cache_dir / "hub"
    if not hub_dir.exists():
        return
    keep_dirs = {f"models--{m.replace('/', '--')}" for m in keep_models}
    removed = []
    for p in hub_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("models--") and p.name not in keep_dirs:
            shutil.rmtree(p)
            removed.append(p.name)
    if removed:
        print("[cache] removed:")
        for n in removed:
            print(f"  - {n}")
    else:
        print("[cache] no unused model cache removed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-model", default=os.getenv("MASK_BERT_MODEL_NAME", "tohoku-nlp/bert-base-japanese-v3"))
    ap.add_argument("--cache-dir", default=os.getenv("HF_HOME", "/workspace/.cache/huggingface"))
    ap.add_argument("--clean-unused", action="store_true", help="Remove other HF model caches")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    required = {args.mask_model}

    # Download/cache mask BERT
    snapshot_with_progress(args.mask_model, cache_dir)

    # Optional cleanup
    if args.clean_unused:
        clean_unused(cache_dir, required)


if __name__ == "__main__":
    main()
