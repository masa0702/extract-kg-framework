import argparse
import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple, Any, Optional


JSONL_RE = re.compile(r"^ont_(\d+)_(.+)_ground_truth\.jsonl$")


def load_exclude_ids(txt_path: Path) -> Set[str]:
    ids = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def is_empty_sent_ja(v: Optional[Any]) -> bool:
    if v is None:
        return True
    if not isinstance(v, str):
        # sent_ja が文字列じゃないのはデータとして怪しいので「空扱い」に寄せる
        return True
    return len(v.strip()) == 0


def filter_jsonl_with_report(
    jsonl_path: Path,
    exclude_ids: Set[str],
    out_path: Path,
    removed_out_path: Path,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    removed_out_path.parent.mkdir(parents=True, exist_ok=True)

    before = 0
    kept = 0
    removed_total = 0
    removed_by_id = 0
    removed_by_empty = 0
    json_error = 0

    with jsonl_path.open("r", encoding="utf-8") as rf, \
         out_path.open("w", encoding="utf-8") as wf, \
         removed_out_path.open("w", encoding="utf-8") as rpf:

        for ln, line in enumerate(rf, start=1):
            s = line.strip()
            if not s:
                continue

            before += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                json_error += 1
                # 壊れ行は除外（レポートに残す）
                removed_total += 1
                rpf.write(json.dumps({
                    "source_file": jsonl_path.name,
                    "line_no": ln,
                    "id": None,
                    "reason": "json_decode_error",
                    "raw": s[:5000],  # 念のため上限
                }, ensure_ascii=False) + "\n")
                continue

            rec_id = obj.get("id")
            sent_ja = obj.get("sent_ja", None)

            # 理由の優先順位：by_id を先に判定して “識別” をブレさせない
            if rec_id in exclude_ids:
                removed_total += 1
                removed_by_id += 1
                rpf.write(json.dumps({
                    "source_file": jsonl_path.name,
                    "line_no": ln,
                    "id": rec_id,
                    "reason": "by_id",
                }, ensure_ascii=False) + "\n")
                continue

            if is_empty_sent_ja(sent_ja):
                removed_total += 1
                removed_by_empty += 1
                rpf.write(json.dumps({
                    "source_file": jsonl_path.name,
                    "line_no": ln,
                    "id": rec_id,
                    "reason": "empty_sent_ja",
                }, ensure_ascii=False) + "\n")
                continue

            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "file": jsonl_path.name,
        "before": before,
        "after": kept,
        "removed_total": removed_total,
        "removed_by_id": removed_by_id,
        "removed_by_empty_sent_ja": removed_by_empty,
        "removed_by_json_error": json_error,
        "delta": kept - before,  # 当然マイナス
        "exclude_ids_count": len(exclude_ids),
        "out_file": str(out_path),
        "removed_records_file": str(removed_out_path),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl_dir",
        default="../../data/T2KGB_JA/all_wikidata_tekgen_ground_truth",
        help="ont_*_ground_truth.jsonl が入っているディレクトリ",
    )
    ap.add_argument(
        "--txt_dir",
        default="../../data/T2KGB_JA/manually_verified_id_text",
        help="selected_ont_*.txt が入っているディレクトリ",
    )
    ap.add_argument(
        "--out_dir",
        default="../../data/T2KGB_JA/target_data",
        help="出力先ディレクトリ",
    )
    ap.add_argument(
        "--suffix",
        default="_target",
        help="出力ファイル名に付けるsuffix（例: _target）",
    )
    args = ap.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    txt_dir = Path(args.txt_dir)
    out_dir = Path(args.out_dir)
    suffix = args.suffix

    if not jsonl_dir.is_dir():
        raise FileNotFoundError(f"jsonl_dir not found: {jsonl_dir}")
    if not txt_dir.is_dir():
        raise FileNotFoundError(f"txt_dir not found: {txt_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    removed_dir = out_dir / "removed_records"
    removed_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    missing_txt = []
    processed_files = 0

    total_before = 0
    total_after = 0
    total_removed = 0
    total_removed_by_id = 0
    total_removed_by_empty = 0
    total_removed_by_json_error = 0

    for jsonl_path in sorted(jsonl_dir.glob("ont_*_ground_truth.jsonl")):
        m = JSONL_RE.match(jsonl_path.name)
        if not m:
            continue

        i = m.group(1)
        category = m.group(2)

        txt_name = f"selected_ont_{i}_{category}.txt"
        txt_path = txt_dir / txt_name
        if not txt_path.exists():
            missing_txt.append(txt_name)
            continue

        exclude_ids = load_exclude_ids(txt_path)

        # 出力ファイル名
        out_name = jsonl_path.name.replace(".jsonl", f"{suffix}.jsonl")
        out_path = out_dir / out_name

        removed_out_name = jsonl_path.name.replace(".jsonl", f"{suffix}_removed.jsonl")
        removed_out_path = removed_dir / removed_out_name

        rep = filter_jsonl_with_report(
            jsonl_path=jsonl_path,
            exclude_ids=exclude_ids,
            out_path=out_path,
            removed_out_path=removed_out_path,
        )
        reports.append(rep)
        processed_files += 1

        total_before += rep["before"]
        total_after += rep["after"]
        total_removed += rep["removed_total"]
        total_removed_by_id += rep["removed_by_id"]
        total_removed_by_empty += rep["removed_by_empty_sent_ja"]
        total_removed_by_json_error += rep["removed_by_json_error"]

        print(
            f"[OK] {jsonl_path.name} "
            f"before={rep['before']} after={rep['after']} "
            f"removed={rep['removed_total']} (by_id={rep['removed_by_id']}, empty_sent_ja={rep['removed_by_empty_sent_ja']}, json_err={rep['removed_by_json_error']})"
        )

    summary = {
        "jsonl_dir": str(jsonl_dir),
        "txt_dir": str(txt_dir),
        "out_dir": str(out_dir),
        "suffix": suffix,
        "processed_files": processed_files,
        "total_before": total_before,
        "total_after": total_after,
        "total_removed": total_removed,
        "total_removed_by_id": total_removed_by_id,
        "total_removed_by_empty_sent_ja": total_removed_by_empty,
        "total_removed_by_json_error": total_removed_by_json_error,
        "total_delta": total_after - total_before,
        "missing_txt_files": missing_txt,
        "per_file_reports": reports,
    }

    report_path = out_dir / f"filter_report{suffix}.json"
    with report_path.open("w", encoding="utf-8") as wf:
        json.dump(summary, wf, ensure_ascii=False, indent=2)

    print("\n==== Summary ====")
    print(f"report: {report_path}")
    print(f"processed_files: {processed_files}")
    print(f"total_before:    {total_before}")
    print(f"total_after:     {total_after}")
    print(f"total_removed:   {total_removed}")
    print(f"  by_id:         {total_removed_by_id}")
    print(f"  empty_sent_ja: {total_removed_by_empty}")
    print(f"  json_error:    {total_removed_by_json_error}")
    if missing_txt:
        print("\n[WARN] 対応するtxtが無くてスキップしたもの:")
        for name in missing_txt:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
