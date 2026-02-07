from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _env_int(name: str, default: int) -> int:
    s = str(os.getenv(name, "")).strip()
    if not s:
        return int(default)
    try:
        return int(s)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    s = str(os.getenv(name, "")).strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


@dataclass
class PromptAcceptRow:
    prompt_id: str
    calls: int = 0
    accepts: int = 0
    skipped_duplicates: int = 0
    cached_hits: int = 0

    @property
    def accept_rate(self) -> float:
        return (self.accepts / self.calls) if self.calls else 0.0


def summarize_prompt_log(prompt_log_path: str) -> Dict[str, Any]:
    per: Dict[str, PromptAcceptRow] = {}
    total_calls = 0
    total_accepts = 0

    for row in _iter_jsonl(prompt_log_path):
        pid = str(row.get("prompt_id", "")).strip()
        if not pid:
            continue

        skipped = bool(row.get("skipped_duplicate"))
        cached = bool(row.get("cached"))

        verdict = _to_int(row.get("final_verdict"))
        if verdict is None:
            verdict = _to_int(row.get("verdict"))
        if verdict is None:
            continue

        st = per.get(pid)
        if st is None:
            st = PromptAcceptRow(prompt_id=pid)
            per[pid] = st
        st.calls += 1
        if skipped:
            st.skipped_duplicates += 1
        if cached:
            st.cached_hits += 1
        if verdict != 0:
            st.accepts += 1

        total_calls += 1
        if verdict != 0:
            total_accepts += 1

    out_rows = []
    for pid, st in sorted(per.items(), key=lambda kv: (kv[0].zfill(4), kv[0])):
        out_rows.append(
            {
                "prompt_id": pid,
                "calls": st.calls,
                "accepts": st.accepts,
                "accept_rate": round(st.accept_rate, 6),
                "skipped_duplicates": st.skipped_duplicates,
                "cached_hits": st.cached_hits,
            }
        )

    return {
        "prompt_log_path": prompt_log_path,
        "total": {
            "calls": total_calls,
            "accepts": total_accepts,
            "accept_rate": round((total_accepts / total_calls) if total_calls else 0.0, 6),
        },
        "by_prompt": out_rows,
    }


def write_prompt_accept_summary(prompt_log_path: str, out_dir: str) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    summary = summarize_prompt_log(prompt_log_path)

    min_calls = _env_int("PROMPT_ACCEPT_WARN_MIN_CALLS", 50)
    warn_thresh = _env_float("PROMPT_ACCEPT_WARN_THRESHOLD", 0.01)
    prompt21_target = _env_float("PROMPT21_ACCEPT_TARGET", 0.10)

    warnings: List[str] = []
    by = summary.get("by_prompt") or []
    for row in by:
        pid = str(row.get("prompt_id", ""))
        calls = int(row.get("calls", 0) or 0)
        rate = float(row.get("accept_rate", 0.0) or 0.0)
        if calls < min_calls:
            continue
        if rate < warn_thresh:
            warnings.append(f"[WARN] prompt accept rate low: prompt_id={pid} calls={calls} rate={rate:.3f} < {warn_thresh:.3f}")
        if pid == "21" and rate < prompt21_target:
            warnings.append(
                f"[WARN] prompt_id=21 accept rate below target: calls={calls} rate={rate:.3f} < target={prompt21_target:.3f}"
            )

    out_path: Optional[str] = None
    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "prompt_accept_summary.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as e:
        warnings.append(f"[WARN] failed to write prompt accept summary: {type(e).__name__}: {e}")
        out_path = None

    return out_path, warnings, summary

