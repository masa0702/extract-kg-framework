import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"invalid json: {path}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                continue
            yield lineno, obj


def _extract_triples(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # select_mode_main.py の出力は extracted_triples。
    # gold は verified_triples のことが多い。
    triples = obj.get("extracted_triples")
    if triples is None:
        triples = obj.get("verified_triples")
    if triples is None:
        triples = obj.get("gold_triples")
    if not triples:
        return []
    if not isinstance(triples, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        sub = normalize_text(t.get("sub"))
        rel = normalize_text(t.get("rel"))
        objv = normalize_text(t.get("obj"))
        if not sub or not rel or not objv:
            continue
        out.append({"sub": sub, "rel": rel, "obj": objv})
    return out


def triple_key(t: Dict[str, Any]) -> Tuple[str, str, str]:
    return (normalize_text(t.get("sub")), normalize_text(t.get("rel")), normalize_text(t.get("obj")))


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def f1(p: float, r: float) -> float:
    return safe_div(2.0 * p * r, p + r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "relation別の pred/gold 件数と、表層一致（sub/rel/objの完全一致）による TP/FP/FN を集計します。"
            "covered は「pred 側で extracted_triples が 1件以上ある文」を対象とします。"
        )
    )
    ap.add_argument("--pred_jsonl", default="", help="pred の extracted_triples.jsonl（select_mode_main.py 出力）")
    ap.add_argument("--gold_jsonl", default="", help="gold の *.jsonl（verified_triples を想定）")
    ap.add_argument(
        "--out_tsv",
        default="",
        help="出力TSV。空なら pred_jsonl と同ディレクトリに *_relation_pr.tsv を作成",
    )
    args = ap.parse_args()

    pred_path = Path(str(args.pred_jsonl))
    gold_path = Path(str(args.gold_jsonl))
    if not str(args.pred_jsonl).strip():
        raise SystemExit("--pred_jsonl を指定してください")
    if not str(args.gold_jsonl).strip():
        raise SystemExit("--gold_jsonl を指定してください")
    if not pred_path.exists():
        raise SystemExit(f"pred_jsonl が見つかりません: {pred_path}")
    if not gold_path.exists():
        raise SystemExit(f"gold_jsonl が見つかりません: {gold_path}")

    out_path = Path(str(args.out_tsv)) if str(args.out_tsv).strip() else pred_path.with_name(pred_path.stem + "_relation_pr.tsv")

    pred_by_id: Dict[str, List[Dict[str, Any]]] = {}
    covered_ids: set[str] = set()
    for _lineno, obj in iter_jsonl(pred_path):
        rid = normalize_text(obj.get("id"))
        if not rid:
            continue
        triples = _extract_triples(obj)
        pred_by_id[rid] = triples
        if triples:
            covered_ids.add(rid)

    gold_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for _lineno, obj in iter_jsonl(gold_path):
        rid = normalize_text(obj.get("id"))
        if not rid:
            continue
        gold_by_id[rid] = _extract_triples(obj)

    # 対象ID集合
    all_ids = set(gold_by_id.keys()) | set(pred_by_id.keys())

    pred_cnt_all: Counter[str] = Counter()
    pred_cnt_cover: Counter[str] = Counter()
    gold_cnt_all: Counter[str] = Counter()
    gold_cnt_cover: Counter[str] = Counter()

    tp_cover: Counter[str] = Counter()
    fp_cover: Counter[str] = Counter()
    fn_cover: Counter[str] = Counter()

    for rid in all_ids:
        preds = pred_by_id.get(rid, [])
        golds = gold_by_id.get(rid, [])

        for t in preds:
            pred_cnt_all[t["rel"]] += 1
            if rid in covered_ids:
                pred_cnt_cover[t["rel"]] += 1

        for t in golds:
            gold_cnt_all[t["rel"]] += 1
            if rid in covered_ids:
                gold_cnt_cover[t["rel"]] += 1

        if rid in covered_ids:
            s_pred = {triple_key(t) for t in preds}
            s_gold = {triple_key(t) for t in golds}
            for sub, rel, objv in (s_pred & s_gold):
                tp_cover[rel] += 1
            for sub, rel, objv in (s_pred - s_gold):
                fp_cover[rel] += 1
            for sub, rel, objv in (s_gold - s_pred):
                fn_cover[rel] += 1

    rels = sorted(set(pred_cnt_all) | set(gold_cnt_all))
    # 解析優先: coveredでのpred数が多い順
    rels.sort(key=lambda r: (pred_cnt_cover.get(r, 0), pred_cnt_all.get(r, 0)), reverse=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "relation",
                "pred_triples_all",
                "pred_triples_covered",
                "gold_triples_all",
                "gold_triples_covered",
                "surface_tp_covered",
                "surface_fp_covered",
                "surface_fn_covered",
                "surface_precision_covered",
                "surface_recall_covered",
                "surface_f1_covered",
                "pred_over_gold_covered",
            ]
        )
        for rel in rels:
            tp = int(tp_cover.get(rel, 0))
            fp = int(fp_cover.get(rel, 0))
            fn = int(fn_cover.get(rel, 0))
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            w.writerow(
                [
                    rel,
                    int(pred_cnt_all.get(rel, 0)),
                    int(pred_cnt_cover.get(rel, 0)),
                    int(gold_cnt_all.get(rel, 0)),
                    int(gold_cnt_cover.get(rel, 0)),
                    tp,
                    fp,
                    fn,
                    p,
                    r,
                    f1(p, r),
                    safe_div(int(pred_cnt_cover.get(rel, 0)), int(gold_cnt_cover.get(rel, 0))),
                ]
            )

    print(f"out: {out_path}")
    print(f"covered_ids: {len(covered_ids)} / all_ids: {len(all_ids)}")


if __name__ == "__main__":
    main()
