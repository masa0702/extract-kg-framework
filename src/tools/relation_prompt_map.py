import argparse
import csv
import json
import re
from typing import Dict, Any, List, Optional


PID_RE = re.compile(r"^P\d+$")


def zpad_prompt_id(strategy_no: str) -> str:
    # "15" -> "15", "1" -> "01" のように2桁に統一（あなたの prompt id に合わせる）
    n = int(str(strategy_no).strip())
    return f"{n:02d}"


def load_prompt_ids(prompt_json_path: str) -> set:
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    return {str(p["id"]).strip() for p in prompts if "id" in p}


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def build_mapping(
    mapping_csv_path: str,
    prompt_json_path: str,
) -> Dict[str, Any]:
    prompt_ids = load_prompt_ids(prompt_json_path)

    by_pid: Dict[str, str] = {}
    by_predicate_ja: Dict[str, str] = {}
    by_ontology_pid: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []

    unknown_prompt_ids: List[Dict[str, Any]] = []

    with open(mapping_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ontology_id = normalize_text(r.get("ontology_id"))
            pid = normalize_text(r.get("pid"))
            predicate_ja = normalize_text(r.get("predicate_ja"))
            strategy_no = normalize_text(r.get("strategy_no"))

            if not strategy_no:
                continue

            prompt_id = zpad_prompt_id(strategy_no)

            # prompt管理JSONに存在しないIDならログに残す（データの整合性チェック）
            if prompt_id not in prompt_ids:
                unknown_prompt_ids.append(
                    {
                        "ontology_id": ontology_id,
                        "pid": pid,
                        "predicate_ja": predicate_ja,
                        "strategy_no": strategy_no,
                        "prompt_id": prompt_id,
                    }
                )

            # インデックス構築
            if pid:
                by_pid[pid] = prompt_id
                if ontology_id:
                    by_ontology_pid[f"{ontology_id}|{pid}"] = prompt_id

            if predicate_ja:
                by_predicate_ja[predicate_ja] = prompt_id

            # 行データ（監査・分析用）
            out_row = dict(r)
            out_row["prompt_id"] = prompt_id
            rows.append(out_row)

    return {
        "by_pid": by_pid,
        "by_predicate_ja": by_predicate_ja,
        "by_ontology_pid": by_ontology_pid,
        "rows": rows,
        "warnings": {
            "unknown_prompt_ids": unknown_prompt_ids
        },
    }


def resolve_prompt_id(
    mapping: Dict[str, Any],
    relation: str,
    *,
    ontology_id: Optional[str] = None
) -> Optional[str]:
    """
    処理中に得られる relation が
    - pid (例: "P50")
    - predicate_ja (例: "著者")
    のどちらでも引けるようにする。
    ontology_id があれば ont-specific を優先。
    """
    rel = normalize_text(relation)
    if not rel:
        return None

    # pidっぽいなら pid で引く
    if PID_RE.match(rel):
        if ontology_id:
            key = f"{normalize_text(ontology_id)}|{rel}"
            hit = mapping.get("by_ontology_pid", {}).get(key)
            if hit:
                return hit
        return mapping.get("by_pid", {}).get(rel)

    # まず日本語ラベル
    hit = mapping.get("by_predicate_ja", {}).get(rel)
    if hit:
        return hit

    # どうしようもない
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_csv", default="../../data/ontology_prompt_alignment.csv")
    ap.add_argument("--prompts_json", default="../../prompts/prompts.json")
    ap.add_argument("--out_json", default="../../prompts/relation_prompt_map.json")
    args = ap.parse_args()

    mapping = build_mapping(args.mapping_csv, args.prompts_json)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 重要な警告だけ表示（unknownがあるなら prompt管理JSONと対応表のズレ）
    unknown = mapping.get("warnings", {}).get("unknown_prompt_ids", [])
    if unknown:
        print(f"[WARN] prompt管理JSONに存在しない prompt_id が {len(unknown)} 件ありました。")
        print("例:", unknown[:3])
    else:
        print("[OK] mapping JSON を生成しました。")


if __name__ == "__main__":
    main()
