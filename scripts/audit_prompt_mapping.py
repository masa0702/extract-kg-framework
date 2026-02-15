#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path("/workspace")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
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


def to_int(x: Any) -> Optional[int]:
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class GroupStat:
    calls: int = 0
    accepts: int = 0

    @property
    def rate(self) -> float:
        return (self.accepts / self.calls) if self.calls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit prompts/relation_prompt_map.json and (optionally) join prompt logs.")
    ap.add_argument(
        "--mapping",
        default=str(REPO_ROOT / "prompts/relation_prompt_map.json"),
        help="relation_prompt_map.json path",
    )
    ap.add_argument("--prompts", default=str(REPO_ROOT / "prompts/prompts.json"), help="prompts.json path")
    ap.add_argument("--prompt-log", default="", help="prompt_log.jsonl path (optional)")
    ap.add_argument("--top-k", type=int, default=30, help="show top-k worst relations by accept rate")
    ap.add_argument("--out", default="", help="write JSON summary to this path (optional)")
    args = ap.parse_args()

    mapping_path = Path(args.mapping)
    prompts_path = Path(args.prompts)
    prompt_log_path = Path(args.prompt_log) if args.prompt_log else None

    mapping = load_json(mapping_path)
    prompts = load_json(prompts_path)
    prompt_ids_defined = {str(p.get("id", "")).strip() for p in (prompts.get("prompts", []) or []) if str(p.get("id", "")).strip()}

    rows = mapping.get("rows") or []
    total_rows = len(rows)

    by_onto = defaultdict(Counter)
    missing_concepts: List[Dict[str, Any]] = []
    missing_prompt_id: List[Dict[str, Any]] = []
    unknown_prompt_id: List[Dict[str, Any]] = []

    for r in rows:
        ont = str(r.get("ontology_id", "")).strip()
        pid = str(r.get("pid", "")).strip()
        pred = str(r.get("predicate_ja", "")).strip()
        pr = str(r.get("prompt_id", "")).strip()
        dc = str(r.get("domain_concept_ja", "")).strip()
        rc = str(r.get("range_concept_ja", "")).strip()

        if ont and pr:
            by_onto[ont][pr] += 1
        if not pr:
            missing_prompt_id.append({"ontology_id": ont, "pid": pid, "predicate_ja": pred})
        elif pr not in prompt_ids_defined:
            unknown_prompt_id.append({"ontology_id": ont, "pid": pid, "predicate_ja": pred, "prompt_id": pr})
        if (not dc) or (not rc):
            missing_concepts.append(
                {
                    "ontology_id": ont,
                    "pid": pid,
                    "predicate_ja": pred,
                    "prompt_id": pr,
                    "domain_concept_ja": dc,
                    "range_concept_ja": rc,
                }
            )

    print("=== mapping summary ===")
    print(f"mapping: {mapping_path}")
    print(f"rows: {total_rows}")
    print(f"missing prompt_id: {len(missing_prompt_id)}")
    print(f"unknown prompt_id: {len(unknown_prompt_id)} (defined prompts: {len(prompt_ids_defined)})")
    print(f"missing domain/range concept: {len(missing_concepts)}")
    print("")
    print("=== prompt_id distribution (by ontology) ===")
    for ont in sorted(by_onto.keys()):
        cnt = by_onto[ont]
        top = ", ".join([f"{k}:{v}" for k, v in cnt.most_common(10)])
        print(f"{ont}: {top}")

    joined = {}
    worst_relations: List[Dict[str, Any]] = []

    if prompt_log_path and prompt_log_path.exists():
        by_rel: Dict[Tuple[str, str, str], GroupStat] = {}
        by_prompt: Dict[str, GroupStat] = {}
        for row in iter_jsonl(prompt_log_path):
            pid = str(row.get("prompt_id", "")).strip()
            ont = str(row.get("ontology_id", "")).strip()
            rel = str(row.get("relation_ja", "")).strip()
            if not pid:
                continue
            verdict = to_int(row.get("final_verdict"))
            if verdict is None:
                verdict = to_int(row.get("verdict"))
            if verdict is None:
                continue

            stp = by_prompt.get(pid)
            if stp is None:
                stp = GroupStat()
                by_prompt[pid] = stp
            stp.calls += 1
            if verdict != 0:
                stp.accepts += 1

            key = (ont, rel, pid)
            st = by_rel.get(key)
            if st is None:
                st = GroupStat()
                by_rel[key] = st
            st.calls += 1
            if verdict != 0:
                st.accepts += 1

        joined["prompt_accept_by_prompt_id"] = {
            pid: {"calls": st.calls, "accepts": st.accepts, "accept_rate": round(st.rate, 6)}
            for pid, st in sorted(by_prompt.items(), key=lambda kv: (kv[0].zfill(4), kv[0]))
        }

        print("")
        print("=== prompt accept (by prompt_id) ===")
        for pid, st in sorted(by_prompt.items(), key=lambda kv: (kv[0].zfill(4), kv[0])):
            print(f"prompt_id={pid} calls={st.calls} accept_rate={st.rate:.3f}")

        # Worst relations among sufficiently-called groups.
        worst = []
        for (ont, rel, pid), st in by_rel.items():
            if st.calls < 30:
                continue
            worst.append((st.rate, st.calls, ont, rel, pid))
        worst.sort(key=lambda t: (t[0], -t[1], t[2], t[3], t[4]))
        worst = worst[: max(1, int(args.top_k))]
        print("")
        print(f"=== worst relations (min_calls=30, top_k={len(worst)}) ===")
        for rate, calls, ont, rel, pid in worst:
            print(f"rate={rate:.3f} calls={calls} ontology={ont} relation={rel} prompt_id={pid}")
            worst_relations.append(
                {
                    "ontology_id": ont,
                    "relation_ja": rel,
                    "prompt_id": pid,
                    "calls": calls,
                    "accept_rate": round(rate, 6),
                }
            )

    out_obj = {
        "mapping_path": str(mapping_path),
        "prompts_path": str(prompts_path),
        "prompt_log_path": str(prompt_log_path) if prompt_log_path else "",
        "rows_total": total_rows,
        "missing_prompt_id": missing_prompt_id,
        "unknown_prompt_id": unknown_prompt_id,
        "missing_concepts": missing_concepts,
        "distribution_by_ontology": {ont: dict(cnt) for ont, cnt in sorted(by_onto.items())},
        "joined": joined,
        "worst_relations": worst_relations,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print("")
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

