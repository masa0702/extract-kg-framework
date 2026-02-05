from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

# Ensure /workspace/src is on sys.path when running as a script.
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules_core.bunsetu import BunsetsuSegmenter, get_nlp


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _segment_sentences_ordered(sentences: List[str]) -> List[List[List[Any]]]:
    nlp = get_nlp()
    docs = nlp.pipe(sentences)
    seg = BunsetsuSegmenter()
    return [seg._segment_doc(doc) for doc in docs]  # type: ignore[arg-type]


def _resolve_sentence(rec: dict[str, Any]) -> str:
    for key in ("sent_ja", "sentence", "sent"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def process_file(input_path: Path, output_path: Path, *, batch_size: int = 32) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        batch_recs: List[dict[str, Any]] = []
        batch_sents: List[str] = []

        def flush() -> None:
            if not batch_recs:
                return
            segs = _segment_sentences_ordered(batch_sents)
            for rec, sent, clauses in zip(batch_recs, batch_sents, segs):
                rec["clauses"] = clauses if sent else []
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            batch_recs.clear()
            batch_sents.clear()

        for rec in _iter_jsonl(input_path):
            sent = _resolve_sentence(rec)
            batch_recs.append(rec)
            batch_sents.append(sent)
            if len(batch_recs) >= batch_size:
                flush()
        flush()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="target_data の文を文節に分割し、sep_bunsetu_target_data に保存する"
    )
    ap.add_argument(
        "--input_dir",
        default="../../data/T2KGB_JA/extract_target_data",
        help="target_data ディレクトリ（JSONL を再帰処理）",
    )
    ap.add_argument(
        "--output_dir",
        default="../../data/T2KGB_JA/sep_bunsetu_target_data",
        help="出力先ディレクトリ（省略時は input_dir の兄弟に sep_bunsetu_target_data を作成）",
    )
    ap.add_argument("--batch_size", type=int, default=32, help="文節分割のバッチサイズ")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.parent / "sep_bunsetu_target_data"

    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"no jsonl files under: {input_dir}")

    print(f"input_dir:  {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"files: {len(jsonl_files)}")

    for fp in jsonl_files:
        rel = fp.relative_to(input_dir)
        out_fp = output_dir / rel
        print(f"processing: {rel}")
        process_file(fp, out_fp, batch_size=args.batch_size)

    print("done")


if __name__ == "__main__":
    main()
