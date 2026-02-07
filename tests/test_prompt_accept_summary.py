import json
import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from modules_core.prompt_monitor import summarize_prompt_log


def test_accept_summary_uses_final_verdict(tmp_path):
    p = tmp_path / "prompt_log.jsonl"
    rows = [
        {"prompt_id": "21", "verdict": 0, "final_verdict": 1, "cached": False},
        {"prompt_id": "22", "verdict": 1, "cached": False, "fallback_from": "21"},
        {"prompt_id": "21", "verdict": 0, "final_verdict": 0, "cached": False},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    summary = summarize_prompt_log(str(p))
    by = {r["prompt_id"]: r for r in summary["by_prompt"]}

    # prompt_id=21: calls=2, accepts=1 (final_verdict=1 is treated as accept)
    assert by["21"]["calls"] == 2
    assert by["21"]["accepts"] == 1

    # prompt_id=22: calls=1, accepts=1
    assert by["22"]["calls"] == 1
    assert by["22"]["accepts"] == 1

