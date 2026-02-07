# -*- coding: utf-8 -*-
"""
eval_wiki_id.py（ローカル完結版の評価スクリプト）

このスクリプトは「一括実行」専用です。実行すると以下を順に行います。

1) ID付与（link）
   - pred の extracted_triples（sub/rel/obj）に対して Wikidata API で QID/PID を付与し、
     `eval/<tag>/with_ids/*.jsonl` を出力します。
   - ネットワークが使えない環境では `--offline`（キャッシュのみ）を使ってください。

2) 全体評価（all）
   - gold と pred を比較して Precision / Recall / F1 を算出します（文字列ベース）。
   - `triple_ids` が両方にあれば IDベース評価も算出します。

3) カバレッジ評価（covered）
   - 2) と同様ですが、pred 側の extracted_triples が空のレコードは母集団から除外します。

想定入出力（pred: select_mode_main.py などの出力）
  {"id": "...", "sent_ja": "...", "extracted_triples":[{"sub":"...","rel":"...","obj":"..."}, ...]}

goldの候補:
  - data/T2KGB_JA/gold_data/*.jsonl （ユーザ作成の gold。存在するならこれを推奨）
  - data/T2KGB_JA/all_wikidata_tekgen_ground_truth/*.jsonl （verified_triples を gold として利用可能）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTO = "auto"


class JaHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "使い方: "
        return super().add_usage(usage, actions, groups, prefix)


class JaRawTextHelpFormatter(argparse.RawTextHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "使い方: "
        return super().add_usage(usage, actions, groups, prefix)


def _is_auto(v: Any) -> bool:
    """
    Treat "", None, and "auto" as "not explicitly specified".
    (Historically some args used default="".)
    """
    if v is None:
        return True
    if isinstance(v, str):
        return (not v.strip()) or (v.strip().lower() == AUTO)
    return False


def _add_japanese_help(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("-h", "--help", action="help", help="このヘルプを表示して終了します。")


def _set_help_titles(ap: argparse.ArgumentParser, *, positional: str = "位置引数", optionals: str = "オプション") -> None:
    # argparse internals, but stable enough for CLI UX improvements
    ap._positionals.title = positional  # type: ignore[attr-defined]
    ap._optionals.title = optionals  # type: ignore[attr-defined]


def _load_env_file_if_missing(path: str, keys: List[str]) -> None:
    """
    Load KEY=VALUE lines from an env file into os.environ only when the key is missing.
    This is intentionally minimal (not a full dotenv parser).
    """
    if not path or (not os.path.exists(path)):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k not in keys:
                    continue
                if k in os.environ and str(os.environ.get(k) or "").strip():
                    continue
                os.environ[k] = v.strip()
    except Exception:
        return


def _guess_latest_results_dir() -> str:
    """
    Prefer results/verX.Y/extract_pred_arg_pair if present, else fallback to results/.
    """
    base = os.path.join(REPO_ROOT, "results")
    if not os.path.isdir(base):
        return base
    best = ""
    best_score = -1.0
    for name in os.listdir(base):
        m = re.match(r"^ver(\d+(?:\.\d+)?)$", name)
        if not m:
            continue
        try:
            score = float(m.group(1))
        except Exception:
            score = -1.0
        cand = os.path.join(base, name, "extract_pred_arg_pair")
        if os.path.isdir(cand) and score >= best_score:
            best = cand
            best_score = score
    return best or base


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _default_eval_root(results_dir: str, eval_tag: str) -> str:
    tag = eval_tag if (eval_tag and eval_tag != "auto") else _now_tag()
    return os.path.join(results_dir, "eval", tag)


def _default_gold_source() -> Tuple[str, str]:
    """
    Return (gold_dir, gold_pattern) with automatic fallback:
    - prefer data/T2KGB_JA/gold_data/*.jsonl (ont_{i}_{category}_gold.jsonl)
    - else use data/T2KGB_JA/all_wikidata_tekgen_ground_truth/*.jsonl (ont_{i}_{category}_ground_truth.jsonl)
    """
    gold_dir = os.path.join(REPO_ROOT, "data/T2KGB_JA/gold_data")
    if os.path.isdir(gold_dir):
        if any(n.endswith(".jsonl") for n in os.listdir(gold_dir)):
            return gold_dir, r"^ont_(\d+)_(.+?)_gold\.jsonl$"
    gt_dir = os.path.join(REPO_ROOT, "data/T2KGB_JA/all_wikidata_tekgen_ground_truth")
    return gt_dir, r"^ont_(\d+)_(.+?)_ground_truth\.jsonl$"


def _default_gold_pattern_for_dir(gold_dir: str) -> str:
    """
    Decide a reasonable default pattern from a gold directory path.
    """
    d = (gold_dir or "").replace("\\", "/").lower()
    if "ground_truth" in d:
        return r"^ont_(\d+)_(.+?)_ground_truth\.jsonl$"
    return r"^ont_(\d+)_(.+?)_gold\.jsonl$"


def _progress(it: Iterable[Any], *, desc: str, disable: bool, total: Optional[int] = None, unit: str = "") -> Iterable[Any]:
    if disable or tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, unit=unit)


def _iter_extracted_triples_files(
    results_dir: str,
    pred_pattern: str,
    *,
    only_prefix: str = "",
    path_contains: str = "",
) -> List[Tuple[str, str, str, str]]:
    """
    Find files matching pred_pattern under results_dir recursively.
    Returns list of (prefix, ont_i, category, path).
    """
    pat = re.compile(pred_pattern)
    out: List[Tuple[str, str, str, str]] = []
    for root, _dirs, files in os.walk(results_dir):
        if path_contains and (path_contains not in root):
            continue
        for name in files:
            if not name.endswith("_extracted_triples.jsonl"):
                continue
            m = pat.match(name)
            if not m:
                continue
            prefix, ont_i, category = m.group(1), m.group(2), m.group(3)
            if only_prefix and prefix != only_prefix:
                continue
            out.append((prefix, ont_i, category, os.path.join(root, name)))
    out.sort()
    return out


def _guess_latest_pred_dir(results_dir: str, pred_pattern: str, *, only_prefix: str = "", path_contains: str = "") -> str:
    """
    Pick the directory that contains the most recently modified extracted_triples file.
    """
    files = _iter_extracted_triples_files(results_dir, pred_pattern, only_prefix=only_prefix, path_contains=path_contains)
    best = ""
    best_mtime = -1.0
    for _prefix, _ont, _cat, path in files:
        try:
            mt = os.path.getmtime(path)
        except Exception:
            continue
        if mt > best_mtime:
            best_mtime = mt
            best = os.path.dirname(path)
    return best


def _guess_latest_matching_dir(results_dir: str, filename_regex: str, *, path_contains: str = "") -> str:
    """
    Pick the directory that contains the most recently modified file whose basename matches filename_regex.
    This does not assume any capture groups.
    """
    pat = re.compile(filename_regex)
    best = ""
    best_mtime = -1.0
    for root, _dirs, files in os.walk(results_dir):
        if path_contains and (path_contains not in root):
            continue
        for name in files:
            if not pat.match(name):
                continue
            path = os.path.join(root, name)
            try:
                mt = os.path.getmtime(path)
            except Exception:
                continue
            if mt > best_mtime:
                best_mtime = mt
                best = root
    return best


# ----------------------------
# Common IO
# ----------------------------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_tsv(path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(_cell(r.get(k, "")) for k in header) + "\n")


def list_files(dir_path: str, pat: re.Pattern[str]) -> List[str]:
    if not os.path.exists(dir_path):
        return []
    names = sorted(os.listdir(dir_path))
    return [os.path.join(dir_path, n) for n in names if pat.match(n)]


def safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


# ----------------------------
# Triple normalization + scoring
# ----------------------------
def norm_text_triple(t: Dict[str, Any]) -> Tuple[str, str, str]:
    return (safe_strip(t.get("sub")), safe_strip(t.get("rel")), safe_strip(t.get("obj")))


def norm_id_triple(ids: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Tuple[str, Optional[str]]]:
    sub_id = ids.get("sub_id")
    rel_id = ids.get("rel_id")
    obj_literal = ids.get("obj_literal")
    obj_id = ids.get("obj_id")
    if obj_literal is not None:
        obj_key = ("LIT", safe_strip(obj_literal))
    else:
        obj_key = ("ID", obj_id)
    return (sub_id, rel_id, obj_key)


def multiset_match(pred_keys: List[Any], gold_keys: List[Any]) -> Tuple[int, int, int]:
    cp = Counter(pred_keys)
    cg = Counter(gold_keys)
    tp = 0
    for k in cp.keys() | cg.keys():
        tp += min(cp.get(k, 0), cg.get(k, 0))
    fp = sum(cp.values()) - tp
    fn = sum(cg.values()) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# ----------------------------
# Wikidata linking (Cell 1)
# ----------------------------
DATE_LIKE_PATTERNS = [
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$"),
    re.compile(r"^\d{4}年$"),
]
NUMBER_LIKE = re.compile(r"^-?\d+(\.\d+)?$")


def is_literal_object(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if NUMBER_LIKE.match(s):
        return True
    for pat in DATE_LIKE_PATTERNS:
        if pat.match(s):
            return True
    return False


class WikidataSearcher:
    def __init__(
        self,
        endpoint: str,
        lang: str,
        fallback_lang: str,
        user_agent: str,
        *,
        sleep_sec: float,
        timeout_sec: int,
        offline: bool,
        cache_path: Optional[str],
    ) -> None:
        if requests is None:  # pragma: no cover
            raise RuntimeError("requests is required for link-ids")
        self.endpoint = endpoint
        self.lang = lang
        self.fallback_lang = fallback_lang
        self.sleep_sec = float(sleep_sec)
        self.timeout_sec = int(timeout_sec)
        self.offline = bool(offline)
        self.cache_path = cache_path

        self.sess = requests.Session()
        # Respect HTTP_PROXY/HTTPS_PROXY/NO_PROXY set in env (including docker/.env if loaded).
        try:
            self.sess.trust_env = True
        except Exception:
            pass
        self.sess.headers.update(
            {
                "User-Agent": user_agent or "eval-wikidata-linker/1.0 (contact: unknown)",
                "Accept": "application/json",
            }
        )

        self.item_cache: Dict[str, Optional[str]] = {}
        self.prop_cache: Dict[str, Optional[str]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path:
            return
        if not os.path.exists(self.cache_path):
            return
        try:
            obj = json.loads(open(self.cache_path, "r", encoding="utf-8").read())
            self.item_cache = obj.get("item_cache", {}) or {}
            self.prop_cache = obj.get("prop_cache", {}) or {}
        except Exception:
            return

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            write_json(
                self.cache_path,
                {"item_cache": self.item_cache, "prop_cache": self.prop_cache},
            )
        except Exception:
            return

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.offline:
            return {}
        # very small retry loop (evaluation tool; keep simple)
        last_err = None
        for i in range(5):
            try:
                r = self.sess.get(self.endpoint, params=params, timeout=self.timeout_sec)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.0 * (2**i))
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (2**i))
        raise RuntimeError(f"Wikidata API request failed: {last_err}")

    def _search_once(self, label: str, entity_type: str, lang: str) -> Optional[str]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": lang,
            "type": entity_type,  # item or property
            "search": label,
            "limit": 1,
            "origin": "*",
        }
        data = self._request(params)
        hits = data.get("search", []) if isinstance(data, dict) else []
        return hits[0].get("id") if hits else None

    def search_entity_id(self, label: str, entity_type: str) -> Optional[str]:
        label = (label or "").strip()
        if not label:
            return None

        cache = self.item_cache if entity_type == "item" else self.prop_cache
        if label in cache:
            return cache[label]

        ent_id = None if self.offline else self._search_once(label, entity_type, self.lang)
        if not self.offline:
            time.sleep(self.sleep_sec)

        if ent_id is None and self.fallback_lang and (not self.offline):
            ent_id = self._search_once(label, entity_type, self.fallback_lang)
            time.sleep(self.sleep_sec)

        cache[label] = ent_id
        self._save_cache()
        return ent_id


def add_ids_to_triple(searcher: WikidataSearcher, triple: Dict[str, Any]) -> Dict[str, Any]:
    sub = safe_strip(triple.get("sub"))
    rel = safe_strip(triple.get("rel"))
    obj = safe_strip(triple.get("obj"))

    sub_id = searcher.search_entity_id(sub, "item") if sub else None
    rel_id = searcher.search_entity_id(rel, "property") if rel else None

    if is_literal_object(obj):
        obj_id = None
        obj_literal = obj if obj else None
    else:
        obj_id = searcher.search_entity_id(obj, "item") if obj else None
        obj_literal = None

    return {"sub_id": sub_id, "rel_id": rel_id, "obj_id": obj_id, "obj_literal": obj_literal}


def cmd_link_ids(args: argparse.Namespace) -> None:
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results_dir not found: {results_dir}")

    out_dir = _default_eval_root(results_dir, args.eval_tag) if _is_auto(args.out_dir) else args.out_dir
    out_with_ids_dir = os.path.join(out_dir, "with_ids")
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_with_ids_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    pred_files = _iter_extracted_triples_files(
        results_dir,
        args.pred_pattern,
        only_prefix=args.only_prefix,
        path_contains=args.path_contains,
    )
    if not pred_files:
        raise SystemExit(f"No pred files found under {results_dir} matching {args.pred_pattern!r}")

    cache_path = os.path.join(logs_dir, "wikidata_link_cache.json")
    searcher = WikidataSearcher(
        endpoint=args.endpoint,
        lang=args.lang,
        fallback_lang=args.fallback_lang,
        user_agent=args.user_agent,
        sleep_sec=args.sleep_sec,
        timeout_sec=args.timeout_sec,
        offline=args.offline,
        cache_path=cache_path,
    )

    miss_detail_path = os.path.join(logs_dir, "missing_ids.detail.jsonl")
    miss_summary_path = os.path.join(logs_dir, "missing_ids.summary.json")
    miss_detail_rows: List[Dict[str, Any]] = []
    summary = defaultdict(
        lambda: {
            "records_in": 0,
            "triples_total": 0,
            "missing_sub_id": 0,
            "missing_rel_id": 0,
            "missing_obj_id": 0,
            "literal_obj": 0,
        }
    )

    no_progress = bool(getattr(args, "no_progress", False))
    for prefix, ont_i, category, path in _progress(pred_files, desc="ID付与: ファイル", disable=no_progress, unit="file"):
        key = os.path.basename(path)
        out_name = f"{prefix}_ont_{ont_i}_{category}_with_ids.jsonl"
        out_path = os.path.join(out_with_ids_dir, out_name)
        out_rows = []
        for rec in _progress(iter_jsonl(path), desc=f"ID付与: レコード ({key})", disable=no_progress, unit="rec"):
            summary[key]["records_in"] += 1
            triples = rec.get("extracted_triples") or []
            if not isinstance(triples, list):
                triples = []
            triple_ids = []
            for t in triples:
                if not isinstance(t, dict):
                    continue
                ids = add_ids_to_triple(searcher, t)
                triple_ids.append(ids)
                summary[key]["triples_total"] += 1
                if ids.get("sub_id") is None:
                    summary[key]["missing_sub_id"] += 1
                if ids.get("rel_id") is None:
                    summary[key]["missing_rel_id"] += 1
                if ids.get("obj_literal") is not None:
                    summary[key]["literal_obj"] += 1
                elif ids.get("obj_id") is None:
                    summary[key]["missing_obj_id"] += 1
                if ids.get("sub_id") is None or ids.get("rel_id") is None or (
                    ids.get("obj_literal") is None and ids.get("obj_id") is None and safe_strip(t.get("obj"))
                ):
                    miss_detail_rows.append(
                        {
                            "file": key,
                            "id": rec.get("id", ""),
                            "triple": t,
                            "triple_ids": ids,
                        }
                    )
            rec2 = dict(rec)
            rec2["triple_ids"] = triple_ids
            out_rows.append(rec2)
        write_jsonl(out_path, out_rows)

    write_jsonl(miss_detail_path, miss_detail_rows)
    write_json(miss_summary_path, summary)
    print(f"with_ids dir: {out_with_ids_dir}")
    print(f"logs dir: {logs_dir}")
    print(f"cache: {cache_path}")
    print(f"missing detail: {miss_detail_path}")
    print(f"missing summary: {miss_summary_path}")


# ----------------------------
# Evaluation (Cell 2/3)
# ----------------------------
@dataclass
class EvalTotals:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _extract_gold_triples(rec: Dict[str, Any], gold_field: str) -> List[Dict[str, Any]]:
    v = rec.get(gold_field)
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return []


def _extract_pred_triples(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = rec.get("extracted_triples")
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return []


def _has_nonempty_pred(rec: Optional[Dict[str, Any]]) -> bool:
    if rec is None:
        return False
    v = rec.get("extracted_triples")
    return isinstance(v, list) and len(v) > 0


def load_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        rid = rec.get("id")
        if rid:
            out[str(rid)] = rec
    return out


def evaluate_pair(
    gold_path: str,
    pred_path: str,
    *,
    gold_field: str,
    covered_only: bool,
) -> Dict[str, Any]:
    gold = load_by_id(gold_path)
    pred = load_by_id(pred_path)

    all_ids = sorted(set(gold.keys()) | set(pred.keys()))
    if covered_only:
        all_ids = [rid for rid in all_ids if _has_nonempty_pred(pred.get(rid))]

    t_text = EvalTotals()
    t_id = EvalTotals()
    id_eval_possible = False
    gold_triples_total = 0

    for rid in all_ids:
        g = gold.get(rid) or {}
        p = pred.get(rid) or {}

        g_tr = _extract_gold_triples(g, gold_field)
        gold_triples_total += len(g_tr)
        p_tr = _extract_pred_triples(p)
        tp, fp, fn = multiset_match([norm_text_triple(x) for x in p_tr], [norm_text_triple(x) for x in g_tr])
        t_text.tp += tp
        t_text.fp += fp
        t_text.fn += fn

        g_ids = g.get("triple_ids")
        p_ids = p.get("triple_ids")
        if isinstance(g_ids, list) and isinstance(p_ids, list):
            id_eval_possible = True
            tp2, fp2, fn2 = multiset_match(
                [norm_id_triple(x) for x in p_ids if isinstance(x, dict)],
                [norm_id_triple(x) for x in g_ids if isinstance(x, dict)],
            )
            t_id.tp += tp2
            t_id.fp += fp2
            t_id.fn += fn2

    p_text, r_text, f1_text = prf(t_text.tp, t_text.fp, t_text.fn)
    out: Dict[str, Any] = {
        "gold_path": gold_path,
        "pred_path": pred_path,
        "gold_field": gold_field,
        "covered_only": covered_only,
        "gold_stats": {
            "eval_records": len(all_ids),
            "gold_triples": gold_triples_total,
        },
        "text": {
            "tp": t_text.tp,
            "fp": t_text.fp,
            "fn": t_text.fn,
            "precision": p_text,
            "recall": r_text,
            "f1": f1_text,
        },
        "id": None,
    }
    if id_eval_possible:
        p_id, r_id, f1_id = prf(t_id.tp, t_id.fp, t_id.fn)
        out["id"] = {
            "tp": t_id.tp,
            "fp": t_id.fp,
            "fn": t_id.fn,
            "precision": p_id,
            "recall": r_id,
            "f1": f1_id,
        }
    return out


def cmd_eval(args: argparse.Namespace, *, covered_only: bool) -> None:
    gold_dir = args.gold_dir
    pred_dir = args.pred_dir
    out_dir = args.out_dir

    if _is_auto(pred_dir):
        results_dir = args.results_dir or _guess_latest_results_dir()
        pred_dir = (
            _guess_latest_matching_dir(results_dir, args.pred_pattern, path_contains=args.path_contains)
            or results_dir
        )
    if _is_auto(out_dir):
        results_dir = args.results_dir or _guess_latest_results_dir()
        out_dir = os.path.join(_default_eval_root(results_dir, args.eval_tag), "single_eval")

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")

    gold_pat = re.compile(args.gold_pattern)
    pred_pat = re.compile(args.pred_pattern)

    gold_files = list_files(gold_dir, gold_pat)
    pred_files = list_files(pred_dir, pred_pat)

    if not gold_files:
        raise SystemExit(f"No gold files: dir={gold_dir}")
    if not pred_files:
        raise SystemExit(f"No pred files: dir={pred_dir}")

    # map (ont_i, category) -> path
    def _key_from(path: str, pat: re.Pattern[str]) -> Optional[Tuple[str, str]]:
        m = pat.match(os.path.basename(path))
        if not m:
            return None
        return (m.group(1), m.group(2))

    gold_map: Dict[Tuple[str, str], str] = {}
    for p in gold_files:
        k = _key_from(p, gold_pat)
        if k:
            gold_map[k] = p
    pred_map: Dict[Tuple[str, str], str] = {}
    for p in pred_files:
        k = _key_from(p, pred_pat)
        if k:
            pred_map[k] = p

    keys = sorted(set(gold_map.keys()) & set(pred_map.keys()))
    if not keys:
        raise SystemExit("No matching (ont_i, category) pairs between gold and pred.")

    results = []
    for k in keys:
        res = evaluate_pair(
            gold_map[k],
            pred_map[k],
            gold_field=args.gold_field,
            covered_only=covered_only,
        )
        res["ont_i"] = k[0]
        res["category"] = k[1]
        results.append(res)

    # micro aggregate (sum of counts)
    ttp = sum(r["text"]["tp"] for r in results)
    tfp = sum(r["text"]["fp"] for r in results)
    tfn = sum(r["text"]["fn"] for r in results)
    mp, mr, mf1 = prf(ttp, tfp, tfn)

    id_present = any(r.get("id") for r in results)
    micro_id = None
    if id_present:
        itp = sum((r["id"]["tp"] if r.get("id") else 0) for r in results)
        ifp = sum((r["id"]["fp"] if r.get("id") else 0) for r in results)
        ifn = sum((r["id"]["fn"] if r.get("id") else 0) for r in results)
        ip, ir, if1 = prf(itp, ifp, ifn)
        micro_id = {"tp": itp, "fp": ifp, "fn": ifn, "precision": ip, "recall": ir, "f1": if1}

    out = {
        "gold_dir": gold_dir,
        "pred_dir": pred_dir,
        "covered_only": covered_only,
        "pairs": len(results),
        "micro_text": {"tp": ttp, "fp": tfp, "fn": tfn, "precision": mp, "recall": mr, "f1": mf1},
        "micro_id": micro_id,
        "results": results,
    }
    write_json(summary_path, out)

    print(f"wrote: {summary_path}")
    print(f"text micro: P={mp:.4f} R={mr:.4f} F1={mf1:.4f}")
    if micro_id:
        print(f"id   micro: P={micro_id['precision']:.4f} R={micro_id['recall']:.4f} F1={micro_id['f1']:.4f}")


# ----------------------------
# Evaluate whole results tree
# ----------------------------
def cmd_eval_results(args: argparse.Namespace) -> None:
    results_dir = args.results_dir
    if _is_auto(results_dir):
        results_dir = _guess_latest_results_dir()
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results_dir not found: {results_dir}")

    gold_dir = args.gold_dir
    gold_pattern = args.gold_pattern
    if _is_auto(gold_dir):
        gold_dir, gold_pattern = _default_gold_source()
    if not os.path.isdir(gold_dir):
        raise SystemExit(f"gold_dir not found: {gold_dir}")
    if _is_auto(gold_pattern):
        gold_pattern = _default_gold_pattern_for_dir(gold_dir)

    eval_tag = args.eval_tag
    out_dir = _default_eval_root(results_dir, eval_tag) if _is_auto(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    pred_with_ids_dir = args.pred_with_ids_dir
    if _is_auto(pred_with_ids_dir) and os.path.isdir(os.path.join(out_dir, "with_ids")):
        pred_with_ids_dir = os.path.join(out_dir, "with_ids")

    gold_pat = re.compile(gold_pattern)
    gold_files = list_files(gold_dir, gold_pat)
    if not gold_files:
        raise SystemExit(f"No gold files found: dir={gold_dir} pattern={gold_pattern!r}")

    def _k_from(path: str, pat: re.Pattern[str]) -> Optional[Tuple[str, str]]:
        m = pat.match(os.path.basename(path))
        if not m:
            return None
        return (m.group(1), m.group(2))

    gold_map: Dict[Tuple[str, str], str] = {}
    for p in gold_files:
        k = _k_from(p, gold_pat)
        if k:
            gold_map[k] = p

    pred_files = _iter_extracted_triples_files(
        results_dir,
        args.pred_pattern,
        only_prefix=args.only_prefix,
        path_contains=args.path_contains,
    )
    if not pred_files:
        raise SystemExit(f"No pred files found under {results_dir} matching {args.pred_pattern!r}")

    # Group by mode prefix (e.g., default / no_verification)
    by_prefix: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    missing_gold: List[Dict[str, Any]] = []

    for prefix, ont_i, category, p in pred_files:
        if (ont_i, category) not in gold_map:
            missing_gold.append({"ont_i": ont_i, "category": category, "pred_path": p, "prefix": prefix})
            continue
        by_prefix[prefix].append((ont_i, category, p))

    index: Dict[str, Any] = {
        "results_dir": results_dir,
        "gold_dir": gold_dir,
        "gold_pattern": gold_pattern,
        "gold_field": args.gold_field,
        "pred_pattern": args.pred_pattern,
        "only_prefix": args.only_prefix,
        "out_dir": out_dir,
        "missing_gold": missing_gold,
        "runs": {},
    }

    no_progress = bool(getattr(args, "no_progress", False))
    for prefix, items in sorted(by_prefix.items()):
        prefix_dir = os.path.join(out_dir, prefix)
        os.makedirs(prefix_dir, exist_ok=True)

        per_file_all: List[Dict[str, Any]] = []
        per_file_cov: List[Dict[str, Any]] = []
        totals_text_all = EvalTotals()
        totals_text_cov = EvalTotals()
        totals_id_all = EvalTotals()
        totals_id_cov = EvalTotals()
        any_id_all = False
        any_id_cov = False
        gold_triples_all = 0
        gold_triples_cov = 0
        eval_records_all = 0
        eval_records_cov = 0

        for ont_i, category, pred_path in _progress(items, desc=f"評価: ファイル ({prefix})", disable=no_progress, unit="file"):
            gold_path = gold_map[(ont_i, category)]
            pred_with_ids_path = ""
            if pred_with_ids_dir:
                cand = os.path.join(pred_with_ids_dir, f"{prefix}_ont_{ont_i}_{category}_with_ids.jsonl")
                if os.path.exists(cand):
                    pred_with_ids_path = cand

            res_all = evaluate_pair(
                gold_path,
                pred_with_ids_path or pred_path,
                gold_field=args.gold_field,
                covered_only=False,
            )
            res_cov = evaluate_pair(
                gold_path,
                pred_with_ids_path or pred_path,
                gold_field=args.gold_field,
                covered_only=True,
            )

            totals_text_all.tp += int(res_all["text"]["tp"])
            totals_text_all.fp += int(res_all["text"]["fp"])
            totals_text_all.fn += int(res_all["text"]["fn"])

            totals_text_cov.tp += int(res_cov["text"]["tp"])
            totals_text_cov.fp += int(res_cov["text"]["fp"])
            totals_text_cov.fn += int(res_cov["text"]["fn"])

            gold_triples_all += int(res_all.get("gold_stats", {}).get("gold_triples", 0))
            gold_triples_cov += int(res_cov.get("gold_stats", {}).get("gold_triples", 0))
            eval_records_all += int(res_all.get("gold_stats", {}).get("eval_records", 0))
            eval_records_cov += int(res_cov.get("gold_stats", {}).get("eval_records", 0))

            if res_all.get("id") is not None:
                any_id_all = True
                totals_id_all.tp += int(res_all["id"]["tp"])
                totals_id_all.fp += int(res_all["id"]["fp"])
                totals_id_all.fn += int(res_all["id"]["fn"])
            if res_cov.get("id") is not None:
                any_id_cov = True
                totals_id_cov.tp += int(res_cov["id"]["tp"])
                totals_id_cov.fp += int(res_cov["id"]["fp"])
                totals_id_cov.fn += int(res_cov["id"]["fn"])

            pred_dirname = os.path.dirname(pred_path)
            prompt_log_guess = os.path.join(pred_dirname, os.path.basename(pred_path).replace("_extracted_triples.jsonl", "_prompt_log.jsonl"))
            meta = {
                "ont_i": ont_i,
                "category": category,
                "pred_path": pred_path,
                "pred_with_ids_path": pred_with_ids_path,
                "gold_path": gold_path,
                "prompt_log_path": (prompt_log_guess if os.path.exists(prompt_log_guess) else ""),
            }
            per_file_all.append({**meta, "eval": res_all})
            per_file_cov.append({**meta, "eval": res_cov})

        p_all, r_all, f1_all = prf(totals_text_all.tp, totals_text_all.fp, totals_text_all.fn)
        p_cov, r_cov, f1_cov = prf(totals_text_cov.tp, totals_text_cov.fp, totals_text_cov.fn)

        micro_id_all = None
        micro_id_cov = None
        if any_id_all:
            ip, ir, if1 = prf(totals_id_all.tp, totals_id_all.fp, totals_id_all.fn)
            micro_id_all = {
                "tp": totals_id_all.tp,
                "fp": totals_id_all.fp,
                "fn": totals_id_all.fn,
                "precision": ip,
                "recall": ir,
                "f1": if1,
            }
        if any_id_cov:
            ip, ir, if1 = prf(totals_id_cov.tp, totals_id_cov.fp, totals_id_cov.fn)
            micro_id_cov = {
                "tp": totals_id_cov.tp,
                "fp": totals_id_cov.fp,
                "fn": totals_id_cov.fn,
                "precision": ip,
                "recall": ir,
                "f1": if1,
            }

        run_summary_all = {
            "prefix": prefix,
            "files": len(items),
            "gold_stats": {
                "eval_records": eval_records_all,
                "gold_triples": gold_triples_all,
            },
            "micro_text": {
                "tp": totals_text_all.tp,
                "fp": totals_text_all.fp,
                "fn": totals_text_all.fn,
                "precision": p_all,
                "recall": r_all,
                "f1": f1_all,
            },
            "micro_id": micro_id_all,
            "per_file": per_file_all,
        }
        run_summary_cov = {
            "prefix": prefix,
            "files": len(items),
            "gold_stats": {
                "eval_records": eval_records_cov,
                "gold_triples": gold_triples_cov,
            },
            "micro_text": {
                "tp": totals_text_cov.tp,
                "fp": totals_text_cov.fp,
                "fn": totals_text_cov.fn,
                "precision": p_cov,
                "recall": r_cov,
                "f1": f1_cov,
            },
            "micro_id": micro_id_cov,
            "per_file": per_file_cov,
        }

        summary_all_path = os.path.join(prefix_dir, "summary_all.json")
        summary_cov_path = os.path.join(prefix_dir, "summary_covered.json")
        write_json(summary_all_path, run_summary_all)
        write_json(summary_cov_path, run_summary_cov)

        tsv_header = [
            "prefix",
            "ont_i",
            "category",
            "eval_records",
            "gold_triples",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "id_tp",
            "id_fp",
            "id_fn",
            "id_precision",
            "id_recall",
            "id_f1",
            "pred_path",
            "pred_with_ids_path",
            "gold_path",
            "prompt_log_path",
        ]
        tsv_rows_all: List[Dict[str, Any]] = [
            {
                "prefix": prefix,
                "ont_i": "micro",
                "category": "micro",
                "eval_records": eval_records_all,
                "gold_triples": gold_triples_all,
                "tp": totals_text_all.tp,
                "fp": totals_text_all.fp,
                "fn": totals_text_all.fn,
                "precision": p_all,
                "recall": r_all,
                "f1": f1_all,
                "id_tp": (micro_id_all or {}).get("tp", ""),
                "id_fp": (micro_id_all or {}).get("fp", ""),
                "id_fn": (micro_id_all or {}).get("fn", ""),
                "id_precision": (micro_id_all or {}).get("precision", ""),
                "id_recall": (micro_id_all or {}).get("recall", ""),
                "id_f1": (micro_id_all or {}).get("f1", ""),
            }
        ]
        for r in per_file_all:
            ev = r["eval"]
            t = ev["text"]
            tid = ev.get("id") or {}
            gstat = ev.get("gold_stats") or {}
            tsv_rows_all.append(
                {
                    "prefix": prefix,
                    "ont_i": r["ont_i"],
                    "category": r["category"],
                    "eval_records": gstat.get("eval_records", ""),
                    "gold_triples": gstat.get("gold_triples", ""),
                    "tp": t["tp"],
                    "fp": t["fp"],
                    "fn": t["fn"],
                    "precision": t["precision"],
                    "recall": t["recall"],
                    "f1": t["f1"],
                    "id_tp": tid.get("tp", ""),
                    "id_fp": tid.get("fp", ""),
                    "id_fn": tid.get("fn", ""),
                    "id_precision": tid.get("precision", ""),
                    "id_recall": tid.get("recall", ""),
                    "id_f1": tid.get("f1", ""),
                    "pred_path": r["pred_path"],
                    "pred_with_ids_path": r["pred_with_ids_path"],
                    "gold_path": r["gold_path"],
                    "prompt_log_path": r["prompt_log_path"],
                }
            )

        tsv_rows_cov: List[Dict[str, Any]] = [
            {
                "prefix": prefix,
                "ont_i": "micro",
                "category": "micro",
                "eval_records": eval_records_cov,
                "gold_triples": gold_triples_cov,
                "tp": totals_text_cov.tp,
                "fp": totals_text_cov.fp,
                "fn": totals_text_cov.fn,
                "precision": p_cov,
                "recall": r_cov,
                "f1": f1_cov,
                "id_tp": (micro_id_cov or {}).get("tp", ""),
                "id_fp": (micro_id_cov or {}).get("fp", ""),
                "id_fn": (micro_id_cov or {}).get("fn", ""),
                "id_precision": (micro_id_cov or {}).get("precision", ""),
                "id_recall": (micro_id_cov or {}).get("recall", ""),
                "id_f1": (micro_id_cov or {}).get("f1", ""),
            }
        ]
        for r in per_file_cov:
            ev = r["eval"]
            t = ev["text"]
            tid = ev.get("id") or {}
            gstat = ev.get("gold_stats") or {}
            tsv_rows_cov.append(
                {
                    "prefix": prefix,
                    "ont_i": r["ont_i"],
                    "category": r["category"],
                    "eval_records": gstat.get("eval_records", ""),
                    "gold_triples": gstat.get("gold_triples", ""),
                    "tp": t["tp"],
                    "fp": t["fp"],
                    "fn": t["fn"],
                    "precision": t["precision"],
                    "recall": t["recall"],
                    "f1": t["f1"],
                    "id_tp": tid.get("tp", ""),
                    "id_fp": tid.get("fp", ""),
                    "id_fn": tid.get("fn", ""),
                    "id_precision": tid.get("precision", ""),
                    "id_recall": tid.get("recall", ""),
                    "id_f1": tid.get("f1", ""),
                    "pred_path": r["pred_path"],
                    "pred_with_ids_path": r["pred_with_ids_path"],
                    "gold_path": r["gold_path"],
                    "prompt_log_path": r["prompt_log_path"],
                }
            )

        tsv_all_path = os.path.join(prefix_dir, "summary_all.tsv")
        tsv_cov_path = os.path.join(prefix_dir, "summary_covered.tsv")
        write_tsv(tsv_all_path, tsv_header, tsv_rows_all)
        write_tsv(tsv_cov_path, tsv_header, tsv_rows_cov)

        print(
            f"micro ({prefix}) all: P={p_all:.4f} R={r_all:.4f} F1={f1_all:.4f} | "
            f"covered: P={p_cov:.4f} R={r_cov:.4f} F1={f1_cov:.4f}"
        )
        index["runs"][prefix] = {
            "summary_all_path": summary_all_path,
            "summary_covered_path": summary_cov_path,
            "tsv_all_path": tsv_all_path,
            "tsv_covered_path": tsv_cov_path,
            "files": len(items),
        }

    write_json(os.path.join(out_dir, "index.json"), index)
    print(f"wrote: {os.path.join(out_dir, 'index.json')}")
    for prefix in sorted(index["runs"].keys()):
        sp1 = index["runs"][prefix]["summary_all_path"]
        sp2 = index["runs"][prefix]["summary_covered_path"]
        tp1 = index["runs"][prefix]["tsv_all_path"]
        tp2 = index["runs"][prefix]["tsv_covered_path"]
        print(f"summary_all ({prefix}): {sp1}")
        print(f"summary_covered ({prefix}): {sp2}")
        print(f"tsv_all ({prefix}): {tp1}")
        print(f"tsv_covered ({prefix}): {tp2}")


def _parse_proxy_host(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    # Very small parser for http://host:port
    s = re.sub(r"^https?://", "", s)
    host = s.split("/", 1)[0].split(":", 1)[0].strip()
    return host


def cmd_check_wikidata(args: argparse.Namespace) -> None:
    if requests is None:
        raise SystemExit("requests is not available in this environment.")

    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    print(f"HTTP_PROXY={http_proxy}")
    print(f"HTTPS_PROXY={https_proxy}")
    print(f"NO_PROXY={no_proxy}")

    proxy_host = _parse_proxy_host(https_proxy) or _parse_proxy_host(http_proxy)
    if proxy_host:
        try:
            socket.getaddrinfo(proxy_host, None)
            print(f"proxy_dns=OK host={proxy_host}")
        except Exception as e:
            print(f"proxy_dns=NG host={proxy_host} error={type(e).__name__}: {e}")

    try:
        socket.getaddrinfo("www.wikidata.org", None)
        print("wikidata_dns=OK host=www.wikidata.org")
    except Exception as e:
        print(f"wikidata_dns=NG host=www.wikidata.org error={type(e).__name__}: {e}")

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "ja",
        "type": "item",
        "search": "東京",
        "limit": 1,
        "origin": "*",
    }
    headers = {"User-Agent": args.user_agent}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"http_status={r.status_code}")
        r.raise_for_status()
        data = r.json()
        hits = data.get("search") or []
        print(f"hits={len(hits)} top_id={(hits[0].get('id') if hits else None)}")
    except Exception as e:
        print(f"request_failed {type(e).__name__}: {e}")


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Wikidata ID付与 + 評価（全体/covered）を一括実行します。",
        add_help=False,
        formatter_class=JaRawTextHelpFormatter,
        epilog=(
            "例:\n"
            "  python src/eval_wiki_id.py\n"
            "  python src/eval_wiki_id.py --eval-tag run_001\n"
            "  python src/eval_wiki_id.py --offline --eval-tag offline_run\n"
            "  python src/eval_wiki_id.py --path-contains 20260207_070440__mode-default --only-prefix default\n"
        ),
    )
    _add_japanese_help(ap)
    _set_help_titles(ap, positional="引数", optionals="オプション")

    ap.add_argument(
        "--results-dir",
        default=_guess_latest_results_dir(),
        help="results ディレクトリ（pred 探索の起点）。",
    )
    ap.add_argument("--eval-tag", default=AUTO, help="出力先 eval/ のタグ名（auto の場合はタイムスタンプ）。")
    ap.add_argument("--out-dir", default=AUTO, help="出力ディレクトリ（auto の場合は <results-dir>/eval/<tag>）。")
    ap.add_argument(
        "--pred-pattern",
        default=r"^(.+?)_ont_(\d+)_(.+?)_extract_target_extracted_triples\.jsonl$",
        help="pred ファイル名の正規表現。(prefix, ont_i, category) をキャプチャしてください。",
    )
    ap.add_argument("--only-prefix", default="", help="指定した prefix（例: default）の pred のみ対象にします。")
    ap.add_argument("--path-contains", default="", help="ディレクトリパスにこの文字列を含む場所だけ探索します。")

    ap.add_argument(
        "--gold-dir",
        default=AUTO,
        help="gold ディレクトリ（auto の場合は data/T2KGB_JA/gold_data を優先し、無ければ ground truth にフォールバック）。",
    )
    ap.add_argument("--gold-pattern", default=AUTO, help="gold ファイル名の正規表現（auto の場合は既定のパターン）。")
    ap.add_argument(
        "--gold-field",
        default="verified_triples",
        help="gold レコード内で triple 配列が入るフィールド名（例: verified_triples / final_triples）。",
    )
    ap.add_argument(
        "--pred-with-ids-dir",
        default=AUTO,
        help="ID評価用 with_ids のディレクトリ。auto の場合は <out_dir>/with_ids があればそれを使います。",
    )

    ap.add_argument("--endpoint", default="https://www.wikidata.org/w/api.php", help="Wikidata API のエンドポイント。")
    ap.add_argument("--lang", default="ja", help="検索に使う言語（wbsearchentities の language）。")
    ap.add_argument("--fallback-lang", default="en", help="見つからない場合のフォールバック言語。空なら無効。")
    ap.add_argument(
        "--user-agent",
        default="MASA-eval-wikidata-linker/1.0 (contact: your_email@example.com)",
        help="Wikidata API に送る User-Agent 文字列（運用時は連絡先入りを推奨）",
    )
    ap.add_argument("--sleep-sec", type=float, default=0.2, help="API 呼び出し間の sleep 秒。")
    ap.add_argument("--timeout-sec", type=int, default=30, help="API リクエストのタイムアウト秒。")
    ap.add_argument("--offline", action="store_true", help="Wikidata API を呼ばず、キャッシュのみで解決します。")
    ap.add_argument("--no-progress", action="store_true", help="進捗バー（tqdm）を表示しません。")

    return ap


def cmd_all(args: argparse.Namespace) -> None:
    results_dir = args.results_dir or _guess_latest_results_dir()
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results_dir not found: {results_dir}")
    args.results_dir = results_dir

    out_dir = _default_eval_root(results_dir, args.eval_tag) if _is_auto(args.out_dir) else args.out_dir
    args.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"out_dir: {out_dir}")

    print("== 1) ID付与 (link-ids)")
    cmd_link_ids(args)

    print("== 2) 評価 (all / covered)")
    cmd_eval_results(args)


def main() -> None:
    # If proxy env vars are not set (e.g., when running outside docker-compose),
    # load them from docker/.env as a convenience.
    _load_env_file_if_missing(
        os.path.join(REPO_ROOT, "docker", ".env"),
        keys=[
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ],
    )
    ap = build_arg_parser()
    args = ap.parse_args()
    cmd_all(args)


if __name__ == "__main__":
    main()
