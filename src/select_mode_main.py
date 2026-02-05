from __future__ import annotations

import os

# Keep CPU thread usage stable across multiprocessing runs (same as src/main.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import queue
import time
from bisect import bisect_left, bisect_right
from datetime import datetime
from glob import glob
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import multiprocessing as mp

import torch
from tqdm.auto import tqdm

from config.filter_settings import PARALLEL_KEYS
from llm.parallel_judge import ParallelJudgeLLMJP
from modules_bert.bert_modules import CKYAnalyzer
from modules_core.bunsetu import DependencyAnalysis
from modules_core.cache_store import SentenceCacheStore
from modules_core.cky_table import CkyTable
from modules_core.match_cache import MatchCacheStore, compute_patterns_fingerprint
from modules_core.matcher import CKYMatcher
from modules_core.ontology_verify import (
    get_ontology_judge,
    get_ontology_resolver,
    load_ontology_relation_aliases,
    normalize_text,
    pick_other_argument,
    prompt_requires_pair,
    render_prompt,
)
from modules_core.pattern_compiler import build_ast_dict, load_and_compile_patterns
from modules_core.text_normalize import strip_trailing_particles
from pattern.pattern_nodes import ParallelNode, VariableNode


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PATTERN_INDEX_JSON_DEFAULT = os.getenv(
    "PATTERN_INDEX_JSON",
    os.path.join(REPO_ROOT, "data/patterns/patterns.index.json"),
)
PATTERN_JSONL_DEFAULT = os.getenv(
    "PATTERN_JSONL",
    os.path.join(REPO_ROOT, "data/patterns/patterns.jsonl"),
)
INPUT_JSONL_DIR_DEFAULT = os.getenv(
    "INPUT_JSONL_DIR",
    (
        os.path.join(REPO_ROOT, "data/T2KGB_JA/extract_target_data")
        if os.path.isdir(os.path.join(REPO_ROOT, "data/T2KGB_JA/extract_target_data"))
        else os.path.join(REPO_ROOT, "data/T2KGB_JA/target_data")
    ),
)
RESULTS_ROOT_DEFAULT = os.getenv(
    "RESULTS_ROOT",
    os.path.join(REPO_ROOT, "results/ver7.0/extract_pred_arg_pair"),
)

PROMPTS_JSON = os.getenv("PROMPTS_JSON", os.path.join(REPO_ROOT, "prompts/prompts.json"))
RELATION_PROMPT_MAP_JSON = os.getenv(
    "RELATION_PROMPT_MAP_JSON",
    os.path.join(REPO_ROOT, "prompts/relation_prompt_map.json"),
)
ONTOLOGY_DIR = os.getenv("ONTOLOGY_DIR", os.path.join(REPO_ROOT, "ontology"))

ONTOLOGY_ID_COL_CANDIDATES = ["ontology_id", "ontology", "ontology_category", "category"]

EXCLUDE_POS = [
    "助詞",
    "接続詞",
    "助動詞",
    "補助記号-句点",
    "補助記号-読点",
    "記号-句点",
    "記号-読点",
]

DEFAULT_GPU_TIMEOUT_SEC = 1800
GPU_TIMEOUT_SEC = int(os.getenv("GPU_TIMEOUT_SEC", str(DEFAULT_GPU_TIMEOUT_SEC)))
GPU_WORKERS = max(1, int(os.getenv("GPU_WORKERS", "1")))

LIT_MAX_FREQ = 200


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def derive_ontology_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("ont_"):
        parts = base.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:3])
    return ""


def _utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _ensure_csv_header(path: str, cols: List[str]) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()


def _append_csv_rows(path: str, cols: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_parallel_variable_groups(ast: Any) -> List[List[str]]:
    groups: List[List[str]] = []

    def visit(node: Any) -> None:
        if isinstance(node, ParallelNode) and hasattr(node, "options"):
            g: List[str] = []
            for opt in node.options or []:
                if isinstance(opt, VariableNode):
                    g.append(f"{opt.symbol}{opt.index}")
            if len(g) >= 2:
                groups.append(g)
        for attr in ("elements", "options", "block"):
            child = getattr(node, attr, None)
            if not child:
                continue
            if isinstance(child, list):
                for c in child:
                    visit(c)
            else:
                visit(child)

    visit(ast)
    return groups


def build_sentence_text_and_offsets(clauses: List[List[Any]]) -> Tuple[str, List[int]]:
    surfaces = [str(cl[0]) for cl in clauses]
    starts = [0]
    total = 0
    for s in surfaces:
        total += len(s)
        starts.append(total)
    return "".join(surfaces), starts


def find_all_occurrences(text: str, sub: str) -> List[int]:
    pos_list: List[int] = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        pos_list.append(idx)
        start = idx + 1
    return pos_list


def count_occurrences_in_span(sorted_positions: List[int], span_start: int, span_end: int) -> int:
    lo = bisect_left(sorted_positions, span_start)
    hi = bisect_left(sorted_positions, span_end)
    return max(0, hi - lo)


def literals_in_order_within_span(
    literals: List[str],
    lit_pos_map: Dict[str, List[int]],
    span_start: int,
    span_end: int,
) -> bool:
    cur = span_start
    for lit in literals:
        pos_list = lit_pos_map.get(lit, [])
        i = bisect_left(pos_list, cur)
        found = False
        while i < len(pos_list):
            p = pos_list[i]
            if p >= span_end:
                break
            if p + len(lit) <= span_end:
                cur = p + len(lit)
                found = True
                break
            i += 1
        if not found:
            return False
    return True


def clean_variable_mapping(varmap: Dict[str, Any], clauses: List[List[Any]]) -> Dict[str, str]:
    new_map: Dict[str, str] = {}
    for var, val in (varmap or {}).items():
        raw_val = str(val) if (val is not None) else ""
        found = None
        for cl in clauses:
            if cl[0] == val:
                found = cl
                break
            if val and isinstance(val, str) and cl[0] in val:
                found = cl
                break
        if not found:
            new_map[str(var)] = strip_trailing_particles(raw_val)
            continue

        # Important: keep spaces inside bunsetsu (e.g., "New York") for Wikidata search.
        # Also, do NOT delete internal particles like "太郎の車".
        surface = str(found[0]) if (found and len(found) > 0) else raw_val
        new_map[str(var)] = strip_trailing_particles(surface, clause=found)
    return new_map


def extract_relation_candidates_from_sentence(sentence: str, ontology_id: str, resolver) -> List[str]:
    if not sentence:
        return []
    ont = normalize_text(ontology_id)
    rows = getattr(resolver, "_rows", [])
    alias_pid = load_ontology_relation_aliases(ONTOLOGY_DIR, ont) if ont else {}
    pid_to_aliases: Dict[str, List[str]] = {}
    for a, pid in (alias_pid or {}).items():
        pid_to_aliases.setdefault(pid, []).append(a)
    out: List[str] = []
    for row in rows:
        if ont and normalize_text(row.get("ontology_id")) != ont:
            continue
        pred = normalize_text(row.get("predicate_ja"))
        if pred and pred in sentence:
            out.append(pred)
            continue
        pid = normalize_text(row.get("pid"))
        if pid:
            for a in pid_to_aliases.get(pid, []):
                if a and (a in sentence):
                    out.append(a)
                    break
    uniq: List[str] = []
    seen: set[str] = set()
    for pred in out:
        if pred in seen:
            continue
        seen.add(pred)
        uniq.append(pred)
    return uniq


def preflight_ontology_llm() -> None:
    if str(os.getenv("ONTOLOGY_VERIFY_PREFLIGHT", "1")).strip().lower() in ("0", "false", "no", ""):
        return
    from llm.llmjp_client import get_llmjp_http_for

    client = get_llmjp_http_for("onto")
    expected = getattr(client, "model", None) or os.getenv("LLMJP_ONTO_MODEL") or os.getenv("LLMJP_MODEL") or "llmjp-13b"
    urls = list(getattr(client, "base_urls", None) or [])
    if not urls:
        urls = [str(getattr(client, "last_base_url", "") or "").strip()] if getattr(client, "last_base_url", None) else []
    urls = [u for u in urls if u]

    if not urls:
        raise RuntimeError("LLMJP_ONTO_BASE_URL(S) が空です")

    for base in urls:
        url = f"{base.rstrip('/')}/models"
        try:
            client.last_base_url = base.rstrip("/")
            client.last_url = url
            r = client._session.get(url, headers=client._headers, timeout=client.timeout_sec)  # type: ignore[attr-defined]
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"onto vLLM の preflight に失敗しました: url={base!r} error={type(e).__name__}: {e}") from e

        ids = set()
        for row in (data or {}).get("data", []) or []:
            mid = (row or {}).get("id")
            if mid:
                ids.add(str(mid))
        if expected and ids and (expected not in ids):
            preview = sorted(ids)
            if len(preview) > 30:
                preview = preview[:30] + ["...(truncated)"]
            raise RuntimeError(f"onto vLLM のモデル名が不整合です: expected={expected!r} url={base!r} available={preview!r}")

    print(f"[preflight] ontology vLLM ok: model={expected!r} urls={urls!r}")


def gpu_worker_loop(worker_id: int, device_id: int, in_q: mp.Queue, out_q: mp.Queue) -> None:
    if torch.cuda.is_available():
        try:
            ndev = int(torch.cuda.device_count() or 0)
        except Exception:
            ndev = 0
        if ndev > 0:
            mapped = int(device_id) % ndev
            torch.cuda.set_device(mapped)

    analyzer = CKYAnalyzer()

    while True:
        task = in_q.get()
        if task is None:
            break
        try:
            cky_table = task["cky_table"]
            t0 = time.time()
            cky_dep = analyzer.analyze_cky_table(cky_table)
            t_analyze = time.time() - t0
            out_q.put(
                {
                    "_worker_id": worker_id,
                    "id": task.get("id", ""),
                    "sentence": task.get("sent", ""),
                    "cky_dep": cky_dep,
                    "clauses": task.get("clauses", []),
                    "t_analyze": t_analyze,
                }
            )
        except Exception as e:
            out_q.put(
                {
                    "_worker_id": worker_id,
                    "id": task.get("id", ""),
                    "sentence": task.get("sent", ""),
                    "_error": str(e),
                }
            )


def run_gpu_stage(
    tasks: List[Dict[str, Any]],
    *,
    log_dir: str,
    prefix: str,
    mode: str,
    timeout_sec: int,
) -> Dict[str, Dict[str, Any]]:
    if not tasks:
        return {}

    os.makedirs(log_dir, exist_ok=True)
    gpu_timing_csv = os.path.join(log_dir, f"{mode}_{prefix}_gpu_timing.csv")
    gpu_timeout_csv = os.path.join(log_dir, f"{mode}_{prefix}_gpu_timeout.csv")
    gpu_error_jsonl = os.path.join(log_dir, f"{mode}_{prefix}_gpu_errors.jsonl")
    if not os.path.exists(gpu_timing_csv):
        with open(gpu_timing_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "t_analyze_sec"])
    if not os.path.exists(gpu_timeout_csv):
        with open(gpu_timeout_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec", "id"])
    if not os.path.exists(gpu_error_jsonl):
        with open(gpu_error_jsonl, "w", encoding="utf-8") as _:
            pass

    start_ts = time.time()
    total = len(tasks)
    results: Dict[str, Dict[str, Any]] = {}

    # Some environments disallow POSIX semaphores (PermissionError) and thus cannot use multiprocessing.Queue/Lock.
    # Fall back to a single-process "GPU stage" to keep the pipeline usable.
    try:
        ctx = mp.get_context("spawn")
        out_q: mp.Queue = ctx.Queue()
        workers: List[Dict[str, Any]] = []
        for wid in range(GPU_WORKERS):
            inq = ctx.Queue(maxsize=1)
            p = ctx.Process(target=gpu_worker_loop, args=(wid, wid, inq, out_q), daemon=True)
            p.start()
            workers.append(
                {"wid": wid, "proc": p, "inq": inq, "busy": False, "start": 0.0, "task": None, "retries": 0}
            )

        idx = 0
        done = 0
        max_retries_per_task = 1

        pbar = tqdm(total=total, desc="GPU stage (match cache)")

        def _any_busy() -> bool:
            return any(w["busy"] for w in workers)

        try:
            while (done < total) or _any_busy():
                for w in workers:
                    if idx >= total:
                        break
                    if w["busy"]:
                        continue
                    task = tasks[idx]
                    w["inq"].put(task)
                    w["busy"] = True
                    w["start"] = time.time()
                    w["task"] = task
                    w["retries"] = 0
                    idx += 1

                while True:
                    try:
                        payload = out_q.get_nowait()
                    except queue.Empty:
                        break
                    wid = payload.get("_worker_id")
                    if wid is None:
                        continue
                    for w in workers:
                        if w["wid"] == wid:
                            w["busy"] = False
                            w["task"] = None
                            break

                    sent = payload.get("sentence", "")
                    if payload.get("_error"):
                        _append_jsonl(
                            gpu_error_jsonl,
                            [{"ts_sec": time.time() - start_ts, "id": payload.get("id", ""), "error": payload.get("_error")}],
                        )
                    else:
                        results[str(sent)] = payload
                        with open(gpu_timing_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([payload.get("id", ""), f'{float(payload.get("t_analyze", 0.0)):.6f}'])

                    done += 1
                    pbar.update(1)

                for w in workers:
                    if not w["busy"]:
                        if not w["proc"].is_alive():
                            try:
                                w["proc"].join(timeout=0.1)
                            except Exception:
                                pass
                            inq = ctx.Queue(maxsize=1)
                            p = ctx.Process(target=gpu_worker_loop, args=(w["wid"], w["wid"], inq, out_q), daemon=True)
                            p.start()
                            w["inq"] = inq
                            w["proc"] = p
                        continue

                    if not w["proc"].is_alive():
                        task = w.get("task") or {}
                        _append_jsonl(
                            gpu_error_jsonl,
                            [{"ts_sec": time.time() - start_ts, "id": task.get("id", ""), "error": "gpu_worker_died"}],
                        )
                        w["busy"] = False
                        w["task"] = None
                        done += 1
                        pbar.update(1)
                        continue

                    if (time.time() - w["start"]) > timeout_sec:
                        task = w.get("task") or {}
                        row_id = task.get("id", "")
                        try:
                            w["proc"].terminate()
                        except Exception:
                            pass
                        try:
                            w["proc"].join(timeout=0.5)
                        except Exception:
                            pass

                        with open(gpu_timeout_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([f"{time.time() - start_ts:.3f}", row_id])

                        inq = ctx.Queue(maxsize=1)
                        p = ctx.Process(target=gpu_worker_loop, args=(w["wid"], w["wid"], inq, out_q), daemon=True)
                        p.start()
                        w["inq"] = inq
                        w["proc"] = p

                        if (task is not None) and (w.get("retries", 0) < max_retries_per_task):
                            w["retries"] = w.get("retries", 0) + 1
                            w["start"] = time.time()
                            w["inq"].put(task)
                        else:
                            w["busy"] = False
                            w["task"] = None
                            done += 1
                            pbar.update(1)

                time.sleep(0.01)
        finally:
            pbar.close()
            for w in workers:
                try:
                    w["inq"].put(None)
                except Exception:
                    pass
            for w in workers:
                try:
                    w["proc"].join(timeout=1.0)
                except Exception:
                    pass
                if w["proc"].is_alive():
                    try:
                        w["proc"].terminate()
                    except Exception:
                        pass

        return results

    except PermissionError as e:
        _append_jsonl(gpu_error_jsonl, [{"ts_sec": 0.0, "id": "", "error": f"mp_disabled: {type(e).__name__}: {e}"}])
        print(f"[WARN] multiprocessing が利用できないため単一プロセスにフォールバックします: {type(e).__name__}: {e}")

    # Single-process fallback.
    analyzer = CKYAnalyzer()
    pbar2 = tqdm(total=total, desc="GPU stage (single-process)")
    for task in tasks:
        row_id = task.get("id", "")
        sent = task.get("sent", "")
        cky_table = task.get("cky_table")
        try:
            t0 = time.time()
            cky_dep = analyzer.analyze_cky_table(cky_table)
            t_analyze = time.time() - t0
            if t_analyze > float(timeout_sec):
                with open(gpu_timeout_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([f"{time.time() - start_ts:.3f}", row_id])
            else:
                results[str(sent)] = {
                    "_worker_id": -1,
                    "id": row_id,
                    "sentence": sent,
                    "cky_dep": cky_dep,
                    "clauses": task.get("clauses", []),
                    "t_analyze": t_analyze,
                }
                with open(gpu_timing_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([row_id, f"{t_analyze:.6f}"])
        except Exception as ex:
            _append_jsonl(
                gpu_error_jsonl,
                [{"ts_sec": time.time() - start_ts, "id": row_id, "error": f"{type(ex).__name__}: {ex}"}],
            )
        finally:
            pbar2.update(1)
    pbar2.close()
    return results


def build_matches_for_sentence(
    sentence: str,
    clauses: List[List[Any]],
    cky_dep: Any,
    ast_dict: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    B = len(clauses)
    if B <= 0:
        return []

    candidate_asts: List[Dict[str, Any]] = []
    for v in range(2, B + 1):
        if v in ast_dict:
            candidate_asts.extend(ast_dict[v])

    sent_text, starts = build_sentence_text_and_offsets(clauses)
    total_len = len(sent_text)
    full_start, full_end = 0, total_len

    par_starts: List[int] = []
    for k in PARALLEL_KEYS:
        if not k:
            continue
        ps = find_all_occurrences(sent_text, k)
        if ps:
            par_starts.extend(ps)
    par_starts.sort()
    parallel_sum_all = len(par_starts)

    uniq_literals: set[str] = set()
    for e in candidate_asts:
        lits = e.get("literal_list", [])
        if lits:
            uniq_literals.update(lits)
    lit_pos_map: Dict[str, List[int]] = {lit: find_all_occurrences(sent_text, lit) for lit in uniq_literals}

    coarse_pass: List[Dict[str, Any]] = []
    for entry in candidate_asts:
        literals = entry.get("literal_list", [])
        par_cnt = int(entry.get("parallel_var_count", 0) or 0)
        if par_cnt >= 2 and parallel_sum_all < (par_cnt - 1):
            continue
        if literals and not literals_in_order_within_span(literals, lit_pos_map, full_start, full_end):
            continue
        coarse_pass.append(entry)
    candidate_asts = coarse_pass

    filtered: List[Dict[str, Any]] = []
    for entry in candidate_asts:
        var_count = int(entry.get("var_count", 0) or 0)
        literals = entry.get("literal_list", [])
        par_cnt = int(entry.get("parallel_var_count", 0) or 0)

        eff_lits: List[str] = []
        if literals:
            for lit in literals:
                freq = len(lit_pos_map.get(lit, []))
                if freq == 0:
                    eff_lits = []
                    break
                if freq <= LIT_MAX_FREQ:
                    eff_lits.append(lit)
            if not eff_lits:
                eff_lits = [max(literals, key=len)]

        cand_ij_base: List[Tuple[int, int]] = []
        if eff_lits:
            first = eff_lits[0]
            first_pos = lit_pos_map.get(first, [])
            for p0 in first_pos:
                cur_end = p0 + len(first)
                ok = True
                for lit in eff_lits[1:]:
                    lst = lit_pos_map.get(lit, [])
                    idx = bisect_left(lst, cur_end)
                    if idx >= len(lst):
                        ok = False
                        break
                    pos = lst[idx]
                    cur_end = pos + len(lit)
                    if cur_end > total_len:
                        ok = False
                        break
                if not ok:
                    continue
                span_start, span_end = p0, cur_end
                i = max(0, bisect_right(starts, span_start) - 1)
                j = max(0, bisect_left(starts, span_end) - 1)
                if j <= i:
                    j = min(B - 1, i + 1)
                cand_ij_base.append((i, j))
        else:
            cand_ij_base = [(0, B - 1)]

        passed = False
        seen_ij: set[Tuple[int, int]] = set()
        for (i, j) in cand_ij_base:
            if (i, j) in seen_ij:
                continue
            seen_ij.add((i, j))
            chunk_num = j - i + 1
            if var_count > chunk_num:
                continue
            if par_cnt >= 2:
                span_start = starts[i]
                span_end = starts[j + 1]
                cnt = count_occurrences_in_span(par_starts, span_start, span_end)
                if cnt < par_cnt - 1:
                    continue
            passed = True
            break

        if not passed:
            continue

        cand_spans = set(cand_ij_base)
        for (i, j) in list(cand_spans):
            if i > 0:
                cand_spans.add((i - 1, j))
            if j < B - 1:
                cand_spans.add((i, j + 1))

        entry["cand_spans"] = sorted(
            [(i + 1, j + 1) for (i, j) in cand_spans if 0 <= i <= j < B],
            key=lambda x: (-(x[1] - x[0]), x[0], x[1]),
        )
        filtered.append(entry)

    matches: List[Dict[str, Any]] = []
    seen = set()

    for entry in filtered:
        ast = entry["ast"]
        matcher = entry.get("matcher") or CKYMatcher(ast, verbose=False)
        par_var_groups = entry.get("parallel_var_groups")
        if par_var_groups is None:
            par_var_groups = extract_parallel_variable_groups(ast)

        for r in matcher.match_table(cky_dep, spans=entry.get("cand_spans")):
            key = frozenset(r.variable_mapping.items())
            if key in seen:
                continue
            seen.add(key)

            varmap_raw = dict(r.variable_mapping)
            varmap_clean = clean_variable_mapping(varmap_raw, clauses)

            Xs: List[Tuple[str, str]] = []
            Ys: List[Tuple[str, str]] = []
            for k, v2 in varmap_clean.items():
                if k.startswith("X"):
                    Xs.append((k, v2))
                elif k.startswith("Y"):
                    Ys.append((k, v2))
            Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
            Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())
            if not Xs or not Ys:
                continue

            x_values = [v for _, v in Xs if v]
            y_values = [v for _, v in Ys if v]
            if not x_values or not y_values:
                continue

            matches.append(
                {
                    "ast_uid": str(entry.get("ast_uid", "")),
                    "pattern_id": str(entry.get("pattern_id", "")),
                    "pattern": str(entry.get("pattern", "")),
                    "var_count": int(entry.get("var_count", 0) or 0),
                    "parallel_var_count": int(entry.get("parallel_var_count", 0) or 0),
                    "literal_list": list(entry.get("literal_list", []) or []),
                    "parallel_var_groups": list(par_var_groups or []),
                    "varmap_raw": varmap_raw,
                    "varmap_clean": varmap_clean,
                    "X_values": x_values,
                    "Y_values": y_values,
                }
            )

    return matches


def canonicalize_relation(resolver, ontology_id: str, rel_raw: str) -> str:
    row = resolver.resolve_relation_row(rel_raw, ontology_id)
    if row:
        return (
            resolver.canonical_relation_label(row, ontology_id)
            or normalize_text(row.get("predicate_ja"))
            or normalize_text(rel_raw)
        )
    return normalize_text(rel_raw)


def resolve_relation_by_partial_match(resolver, ontology_id: str, rel_raw: str) -> Optional[str]:
    """
    no_verification 用:
    - パターンの Y 表層を、ontology に定義された relation label/alias に「部分一致」で吸収する
    - 一致できない（または曖昧）場合は None を返す（= triple を作らない）
    """
    rel_raw_n = normalize_text(rel_raw)
    ont = normalize_text(ontology_id)
    if not (rel_raw_n and ont):
        return None

    # Fast path: resolver が直接解決できるならそれを優先。
    row = resolver.resolve_relation_row(rel_raw_n, ont)
    if row:
        out = resolver.canonical_relation_label(row, ont) or normalize_text(row.get("predicate_ja"))
        return out or None

    alias_pid = load_ontology_relation_aliases(ONTOLOGY_DIR, ont) or {}
    best_pid: Optional[str] = None
    best_score = -1.0
    second_score = -1.0

    for alias, pid in alias_pid.items():
        a = normalize_text(alias)
        p = normalize_text(pid)
        if not (a and p):
            continue
        if (a in rel_raw_n) or (rel_raw_n in a):
            # Prefer tighter/longer match.
            shorter = min(len(a), len(rel_raw_n))
            longer = max(len(a), len(rel_raw_n))
            score = 1.0 if a == rel_raw_n else (shorter / float(longer)) if longer > 0 else 0.0
            if score > best_score:
                second_score = best_score
                best_score = score
                best_pid = p
            elif score > second_score:
                second_score = score

    if best_pid is None:
        return None
    # Avoid accidental ties/ambiguity.
    if (best_score - second_score) < 0.01:
        return None

    # Resolve via PID to get canonical label and prompt mapping row.
    row2 = resolver.resolve_relation_row(best_pid, ont) or resolver.resolve_relation_row(rel_raw_n, ont)
    if not row2:
        return None
    out2 = resolver.canonical_relation_label(row2, ont) or normalize_text(row2.get("predicate_ja"))
    return out2 or None


def _is_parallel_pair(a: str, b: str, parallel_value_groups: List[List[str]]) -> bool:
    if not a or not b:
        return False
    for g in parallel_value_groups:
        s = set(g)
        if a in s and b in s:
            return True
    return False


def _iter_allowed_pairs(x_values: List[str], parallel_value_groups: List[List[str]]):
    for a1, a2 in combinations(x_values, 2):
        if _is_parallel_pair(a1, a2, parallel_value_groups):
            continue
        yield a1, a2


def process_jsonl_select_mode(
    input_jsonl_path: str,
    ast_dict: Dict[int, List[Dict[str, Any]]],
    patterns_fingerprint: str,
    args: argparse.Namespace,
) -> None:
    filename = os.path.basename(input_jsonl_path)
    prefix = filename[:-6] if filename.endswith(".jsonl") else os.path.splitext(filename)[0]

    # Keep input directory structure under results_root so ontology/category folders can be mirrored.
    # Example:
    #   input_jsonl_dir = data/.../extract_target_data
    #   input_jsonl_path = data/.../extract_target_data/ont_1_movie/foo.jsonl
    # => results_root/extract_target_data/ont_1_movie/foo/select_mode/<run_tag>/...
    input_root_name = os.path.basename(os.path.abspath(str(args.input_jsonl_dir)).rstrip("/"))
    try:
        rel_path = os.path.relpath(input_jsonl_path, args.input_jsonl_dir)
    except Exception:
        rel_path = os.path.basename(input_jsonl_path)
    rel_dir = os.path.dirname(rel_path)
    if rel_dir in ("", "."):
        rel_dir = ""
    # Safety: if input_jsonl_path is outside input_jsonl_dir, fall back to non-recursive behavior.
    if rel_path.startswith(".."):
        rel_dir = os.path.basename(os.path.dirname(input_jsonl_path))

    base_out_dir = os.path.join(args.results_root, input_root_name, rel_dir, prefix)
    cache_dir = os.path.join(base_out_dir, "cache")
    run_dir = os.path.join(base_out_dir, "select_mode", args.run_tag)
    run_log_dir = os.path.join(run_dir, "logs")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    print(f"入力JSONL: {input_jsonl_path}")
    print(f"保存先(run): {run_dir}")
    print(f"cache: {cache_dir}")

    records = list(iter_jsonl(input_jsonl_path))
    if not records:
        print("JSONLが空です。スキップします。")
        return

    sentences: List[str] = []
    seen_sentences: set[str] = set()
    for r in records:
        s = r.get("sent_ja") or r.get("sent") or ""
        if s and s not in seen_sentences:
            seen_sentences.add(s)
            sentences.append(s)

    default_ontology_id = os.getenv("DEFAULT_ONTOLOGY_ID", "")
    file_ontology_id = derive_ontology_id_from_filename(input_jsonl_path)

    dep_cache = SentenceCacheStore(os.path.join(cache_dir, "dep"), "dep")
    cky_cache = SentenceCacheStore(os.path.join(cache_dir, "cky"), "cky")

    dep_data = dep_cache.load_many(sentences)
    new_sentences = [s for s in sentences if s not in dep_data]

    depana = DependencyAnalysis()
    cky_obj = CkyTable()

    if new_sentences:
        print(f"GiNZA 解析: {len(new_sentences)} 文")
        dep_results = depana.analyze_sentences(new_sentences)
        for sent, data in dep_results.items():
            dep_cache.save(sent, data)
        dep_data.update(dep_results)

    cky_data = cky_cache.load_many(sentences)
    new_cky = [s for s in sentences if s not in cky_data]
    for s in new_cky:
        dep_entry = dep_data.get(s)
        if not dep_entry:
            continue
        cky_entry = cky_obj.build_entry_from_clauses(dep_entry.get("clauses", []))
        cky_cache.save(s, cky_entry)
        cky_data[s] = cky_entry

    record_rows: List[Dict[str, Any]] = []
    for r in records:
        s = r.get("sent_ja") or r.get("sent") or ""
        if not s:
            continue
        info = cky_data.get(s)
        if not info:
            continue
        ontology_id = ""
        for col in ONTOLOGY_ID_COL_CANDIDATES:
            val = r.get(col)
            if val is not None and str(val).strip() != "":
                ontology_id = str(val)
                break
        if not ontology_id:
            ontology_id = default_ontology_id or file_ontology_id
        record_rows.append(
            {
                "id": r.get("id", ""),
                "sentence": s,
                "clauses": info.get("clauses", []),
                "cky_table": info.get("dependency_table", []),
                "ontology_id": ontology_id,
            }
        )

    if not record_rows:
        print("CKY情報のある文がありません。スキップします。")
        return

    # Prepare output paths (run-scoped).
    candidate_csv = os.path.join(run_dir, f"{prefix}_triples_candidate.csv")
    candidate_slim_csv = os.path.join(run_dir, f"{prefix}_triples_candidate_slim.csv")
    verified_csv = os.path.join(run_dir, f"{args.mode}_{prefix}_triples_verified.csv")
    vis_csv = os.path.join(run_dir, f"{args.mode}_{prefix}_ast_visualization.csv")
    prompt_log_path = os.path.join(run_dir, f"{args.mode}_{prefix}_prompt_log.jsonl")
    extracted_jsonl_path = os.path.join(run_dir, f"{args.mode}_{prefix}_extracted_triples.jsonl")
    parallel_log_path = os.path.join(run_log_dir, f"{args.mode}_{prefix}_parallel_verify.jsonl")

    triple_cols = [
        "id",
        "sentence",
        "ontology_id",
        "relation_ja",
        "pid",
        "prompt_id",
        "prompt_name",
        "domain_arg",
        "range_arg",
        "domain_concept_ja",
        "range_concept_ja",
        "verdict",
        "ast_uid",
        "stage",
    ]
    slim_cols = ["id", "sentence", "ontology_id", "relation_ja", "domain_arg", "range_arg", "ast_uid", "stage"]
    vis_cols = [
        "id",
        "sentence",
        "ast_uid",
        "pattern_id",
        "pattern",
        "var_count",
        "parallel_var_count",
        "literals",
        "X_vars",
        "Y_vars",
        "varmap_raw",
        "varmap_clean",
        "parallel_var_names",
        "parallel_elements",
    ]
    if args.export_ast_repr:
        vis_cols.append("ast_repr")

    _ensure_csv_header(candidate_csv, triple_cols)
    if args.export_candidate_slim:
        _ensure_csv_header(candidate_slim_csv, slim_cols)
    _ensure_csv_header(verified_csv, triple_cols)
    _ensure_csv_header(vis_csv, vis_cols)
    if not os.path.exists(prompt_log_path):
        with open(prompt_log_path, "w", encoding="utf-8") as _:
            pass
    if not os.path.exists(extracted_jsonl_path):
        with open(extracted_jsonl_path, "w", encoding="utf-8") as _:
            pass
    if not os.path.exists(parallel_log_path):
        with open(parallel_log_path, "w", encoding="utf-8") as _:
            pass

    # Build unique sentence tasks for match-cache generation.
    unique_by_sentence: Dict[str, Dict[str, Any]] = {}
    for rr in record_rows:
        s = rr["sentence"]
        if s not in unique_by_sentence:
            unique_by_sentence[s] = {
                "id": _sha1_12(s),
                "sent": s,
                "clauses": rr.get("clauses", []),
                "cky_table": rr.get("cky_table", []),
            }

    match_cache_root = os.path.join(cache_dir, "match")
    match_store = MatchCacheStore.from_dir(match_cache_root, patterns_fingerprint)

    existing = {} if (args.cache_mode == "refresh") else match_store.load_many(unique_by_sentence.keys())
    to_build: List[Dict[str, Any]] = []
    if args.cache_mode == "refresh":
        to_build = list(unique_by_sentence.values())
    else:
        for s, payload in unique_by_sentence.items():
            if s not in existing:
                to_build.append(payload)

    if to_build:
        print(f"matchキャッシュ生成: {len(to_build)} 文 (cache_mode={args.cache_mode})")
        gpu_results = run_gpu_stage(to_build, log_dir=run_log_dir, prefix=prefix, mode=args.mode, timeout_sec=GPU_TIMEOUT_SEC)
        ok = 0
        for task in tqdm(to_build, desc="Match stage (cache)"):
            sent = task.get("sent", "")
            payload = gpu_results.get(sent)
            if not payload or payload.get("_error"):
                continue
            matches = build_matches_for_sentence(
                sent,
                payload.get("clauses", []),
                payload.get("cky_dep"),
                ast_dict,
            )
            match_store.save(sent, matches)
            existing[sent] = match_store.load(sent) or {"matches": matches}
            ok += 1
        print(f"matchキャッシュ生成 完了: {ok}/{len(to_build)} 文")

    # Candidate/vis generation is always done from match cache.
    ontology_resolver = get_ontology_resolver(RELATION_PROMPT_MAP_JSON, PROMPTS_JSON, ONTOLOGY_DIR)
    rel_label_cache: Dict[Tuple[str, str], str] = {}

    cand_rows_total = 0
    vis_rows_total = 0

    os.makedirs(os.path.dirname(candidate_csv), exist_ok=True)
    with (
        open(candidate_csv, "a", newline="", encoding="utf-8") as fcand,
        open(vis_csv, "a", newline="", encoding="utf-8") as fvis,
    ):
        wcand = csv.DictWriter(fcand, fieldnames=triple_cols, extrasaction="ignore")
        wvis = csv.DictWriter(fvis, fieldnames=vis_cols, extrasaction="ignore")
        if args.export_candidate_slim:
            fslim = open(candidate_slim_csv, "a", newline="", encoding="utf-8")
            wslim = csv.DictWriter(fslim, fieldnames=slim_cols, extrasaction="ignore")
        else:
            fslim = None
            wslim = None

        try:
            for rr in tqdm(record_rows, desc="candidate/vis"):
                sent_id = rr.get("id", "")
                sentence = rr.get("sentence", "")
                ontology_id = rr.get("ontology_id", "") or ""

                cache_payload = existing.get(sentence)
                if not cache_payload:
                    continue
                matches = cache_payload.get("matches") or []

                for m in matches:
                    ast_uid = m.get("ast_uid", "")
                    x_values = list(m.get("X_values", []) or [])
                    y_values = list(m.get("Y_values", []) or [])

                    for xv, yv in product(x_values, y_values):
                        key = (ontology_id, yv)
                        rel_out = rel_label_cache.get(key)
                        if rel_out is None:
                            rel_out = canonicalize_relation(ontology_resolver, ontology_id, yv)
                            rel_label_cache[key] = rel_out
                        row = {
                            "id": sent_id,
                            "sentence": sentence,
                            "ontology_id": ontology_id,
                            "relation_ja": rel_out,
                            "pid": "",
                            "prompt_id": "",
                            "prompt_name": "",
                            "domain_arg": xv,
                            "range_arg": "",
                            "domain_concept_ja": "",
                            "range_concept_ja": "",
                            "verdict": "",
                            "ast_uid": ast_uid,
                            "stage": "candidate",
                        }
                        wcand.writerow(row)
                        cand_rows_total += 1
                        if wslim is not None:
                            wslim.writerow({c: row.get(c, "") for c in slim_cols})

                    varmap_raw = m.get("varmap_raw", {}) or {}
                    varmap_clean = m.get("varmap_clean", {}) or {}

                    Xs: List[Tuple[str, str]] = []
                    Ys: List[Tuple[str, str]] = []
                    for k, v2 in varmap_clean.items():
                        if str(k).startswith("X"):
                            Xs.append((str(k), str(v2)))
                        elif str(k).startswith("Y"):
                            Ys.append((str(k), str(v2)))
                    Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
                    Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())

                    par_groups = m.get("parallel_var_groups", []) or []
                    par_names: List[str] = []
                    for g in par_groups:
                        par_names.extend(list(g))
                    par_elems = [varmap_clean.get(name, "") for name in par_names if name in varmap_clean]

                    vis_row: Dict[str, Any] = {
                        "id": sent_id,
                        "sentence": sentence,
                        "ast_uid": ast_uid,
                        "pattern_id": m.get("pattern_id", ""),
                        "pattern": m.get("pattern", ""),
                        "var_count": m.get("var_count", 0),
                        "parallel_var_count": m.get("parallel_var_count", 0),
                        "literals": "|".join(m.get("literal_list", []) or []),
                        "X_vars": "|".join([f"{k}:{v}" for k, v in Xs]) if Xs else "",
                        "Y_vars": "|".join([f"{k}:{v}" for k, v in Ys]) if Ys else "",
                        "varmap_raw": json.dumps(varmap_raw, ensure_ascii=False),
                        "varmap_clean": json.dumps(varmap_clean, ensure_ascii=False),
                        "parallel_var_names": json.dumps(par_names, ensure_ascii=False),
                        "parallel_elements": json.dumps(par_elems, ensure_ascii=False),
                    }
                    if args.export_ast_repr:
                        vis_row["ast_repr"] = ""
                    wvis.writerow(vis_row)
                    vis_rows_total += 1
        finally:
            if fslim is not None:
                fslim.close()

    print(f"candidate rows: {cand_rows_total}  vis rows: {vis_rows_total}")

    # Verification stage (mode-dependent).
    mode = args.mode

    judge_parallel = ParallelJudgeLLMJP()
    if mode == "no_verification":
        # Parallel verification is still executed for logging, but it does not affect extraction here.
        for rr in tqdm(record_rows, desc="parallel(no_verification)"):
            sent_id = rr.get("id", "")
            sentence = rr.get("sentence", "")
            ontology_id = rr.get("ontology_id", "") or ""
            cache_payload = existing.get(sentence)
            if not cache_payload:
                continue
            for m in cache_payload.get("matches") or []:
                ast_uid = m.get("ast_uid", "")
                varmap_clean = m.get("varmap_clean", {}) or {}
                for g in m.get("parallel_var_groups", []) or []:
                    vals = [str(varmap_clean.get(name, "") or "") for name in g]
                    vals = [v for v in vals if v]
                    if len(vals) < 2:
                        continue
                    res = judge_parallel.judge_parallel(sentence, vals)
                    _append_jsonl(
                        parallel_log_path,
                        [
                            {
                                "id": sent_id,
                                "sentence": sentence,
                                "ontology_id": ontology_id,
                                "ast_uid": ast_uid,
                                "pattern_id": m.get("pattern_id", ""),
                                "group": g,
                                "values": vals,
                                "result": res,
                            }
                        ],
                    )

        # Extract triples from (X, Y) combinations:
        # - Require relation to be absorbed into ontology-defined one by partial match; otherwise skip.
        # - For each unordered (arg1,arg2) pair, emit both directions:
        #   (arg1, rel, arg2) and (arg2, rel, arg1)
        for rr in tqdm(record_rows, desc="extracted_triples(no_verification)"):
            sent_id = rr.get("id", "")
            sentence = rr.get("sentence", "")
            ontology_id = rr.get("ontology_id", "") or ""
            cache_payload = existing.get(sentence)
            if not cache_payload:
                _append_jsonl(extracted_jsonl_path, [{"id": sent_id, "sent_ja": sentence, "extracted_triples": []}])
                continue
            matches = cache_payload.get("matches") or []

            triples_out: List[Dict[str, str]] = []
            seen_tr_set: set[Tuple[str, str, str]] = set()

            for m in matches:
                x_values = list(m.get("X_values", []) or [])
                y_values = list(m.get("Y_values", []) or [])
                if len(x_values) < 2:
                    continue
                for yv in y_values:
                    rel = resolve_relation_by_partial_match(ontology_resolver, ontology_id, yv)
                    if not rel:
                        continue
                    for a, b in combinations(x_values, 2):
                        for sub, obj in ((a, b), (b, a)):
                            if not (sub and obj):
                                continue
                            key3 = (sub, rel, obj)
                            if key3 in seen_tr_set:
                                continue
                            seen_tr_set.add(key3)
                            triples_out.append({"sub": sub, "rel": rel, "obj": obj})

            _append_jsonl(extracted_jsonl_path, [{"id": sent_id, "sent_ja": sentence, "extracted_triples": triples_out}])

        print("no_verification: verified は生成しません（0行）。extracted_triples は match 由来で生成します。")
        return

    if mode in ("default", "no_parallel_verification"):
        preflight_ontology_llm()

    ontology_judge = get_ontology_judge()
    log_dup_skips = str(os.getenv("LOG_DUPLICATE_SKIPS", "")).lower() in ("1", "true", "yes")

    verified_rows_total = 0

    with open(verified_csv, "a", newline="", encoding="utf-8") as fver:
        wver = csv.DictWriter(fver, fieldnames=triple_cols, extrasaction="ignore")

        for rr in tqdm(record_rows, desc=f"verified({mode})"):
            sent_id = rr.get("id", "")
            sentence = rr.get("sentence", "")
            ontology_id = rr.get("ontology_id", "") or ""

            cache_payload = existing.get(sentence)
            if not cache_payload:
                _append_jsonl(extracted_jsonl_path, [{"id": sent_id, "sent_ja": sentence, "extracted_triples": []}])
                continue

            matches = cache_payload.get("matches") or []

            seen_triples: set[Tuple[str, str, str, str, str]] = set()
            seen_prompt_calls: set[Tuple[str, str, str, str, str]] = set()
            rel_row_cache: Dict[str, Dict[str, Any]] = {}

            verified_rows: List[Dict[str, Any]] = []
            triples_out: List[Dict[str, str]] = []
            seen_tr_set: set[Tuple[str, str, str]] = set()

            for m in matches:
                ast_uid = m.get("ast_uid", "")
                x_values = list(m.get("X_values", []) or [])
                y_values = list(m.get("Y_values", []) or [])
                if (not x_values) or (not y_values):
                    continue

                parallel_value_groups: List[List[str]] = []
                if mode == "default":
                    ok_parallel = True
                    varmap_clean = m.get("varmap_clean", {}) or {}
                    for g in m.get("parallel_var_groups", []) or []:
                        vals = [str(varmap_clean.get(name, "") or "") for name in g]
                        vals = [v for v in vals if v]
                        if len(vals) < 2:
                            continue
                        res = judge_parallel.judge_parallel(sentence, vals)
                        _append_jsonl(
                            parallel_log_path,
                            [
                                {
                                    "id": sent_id,
                                    "sentence": sentence,
                                    "ontology_id": ontology_id,
                                    "ast_uid": ast_uid,
                                    "pattern_id": m.get("pattern_id", ""),
                                    "group": g,
                                    "values": vals,
                                    "result": res,
                                }
                            ],
                        )
                        if res is False:
                            ok_parallel = False
                            break
                        parallel_value_groups.append(vals)
                    if not ok_parallel:
                        continue

                elif mode == "no_parallel_verification":
                    pass

                resolved_rows: Dict[str, Dict[str, Any]] = {}
                for rel in y_values:
                    row = rel_row_cache.get(rel) or ontology_resolver.resolve_relation_row(rel, ontology_id)
                    if row:
                        rel_row_cache[rel] = row
                        resolved_rows[rel] = row
                if not resolved_rows:
                    for rel in extract_relation_candidates_from_sentence(sentence, ontology_id, ontology_resolver):
                        row = rel_row_cache.get(rel) or ontology_resolver.resolve_relation_row(rel, ontology_id)
                        if row:
                            rel_row_cache[rel] = row
                            resolved_rows[rel] = row

                for rel_raw, row in resolved_rows.items():
                    rel_canon = (
                        ontology_resolver.canonical_relation_label(row, ontology_id)
                        or normalize_text(row.get("predicate_ja"))
                        or normalize_text(rel_raw)
                    )
                    prompt_id = str(row.get("prompt_id", "") or "")
                    prompt = ontology_resolver.get_prompt(prompt_id)
                    if not prompt:
                        continue

                    domain_concept, range_concept = ontology_resolver.resolve_concepts(row, ontology_id)
                    domain_concept = domain_concept or ""
                    range_concept = range_concept or ""
                    pid = str(row.get("pid", "") or "")
                    prompt_name = prompt.prompt_name

                    if prompt_requires_pair(prompt):
                        if len(x_values) < 2:
                            continue
                        if not domain_concept or not range_concept:
                            continue

                        for arg1, arg2 in _iter_allowed_pairs(x_values, parallel_value_groups):
                            call_key = (ontology_id, rel_canon, prompt_id, arg1, arg2)
                            if call_key in seen_prompt_calls:
                                if log_dup_skips:
                                    _append_jsonl(
                                        prompt_log_path,
                                        [
                                            {
                                                "id": sent_id,
                                                "relation_ja": rel_canon,
                                                "relation_raw": rel_raw,
                                                "ontology_id": ontology_id,
                                                "prompt_id": prompt_id,
                                                "prompt_name": prompt_name,
                                                "mode": "pair",
                                                "arg1": arg1,
                                                "arg2": arg2,
                                                "verdict": None,
                                                "skipped_duplicate": True,
                                            }
                                        ],
                                    )
                                continue
                            seen_prompt_calls.add(call_key)

                            prompt_text = render_prompt(
                                prompt,
                                {
                                    "relation_ja": rel_canon,
                                    "domain_concept_ja": domain_concept,
                                    "range_concept_ja": range_concept,
                                    "arg1": arg1,
                                    "arg2": arg2,
                                    "context_sentence": sentence,
                                },
                            )
                            try:
                                verdict = ontology_judge.judge_prompt(prompt_text)
                            except Exception as e:
                                raise RuntimeError(
                                    f"ontology verify failed: id={sent_id!r} prompt_id={prompt_id!r} error={e}"
                                ) from e
                            meta = getattr(ontology_judge, "last_meta", {}) or {}
                            err = getattr(ontology_judge, "last_error", None)
                            row_log: Dict[str, Any] = {
                                "id": sent_id,
                                "relation_ja": rel_canon,
                                "relation_raw": rel_raw,
                                "ontology_id": ontology_id,
                                "prompt_id": prompt_id,
                                "prompt_name": prompt_name,
                                "mode": "pair",
                                "arg1": arg1,
                                "arg2": arg2,
                                "verdict": verdict,
                                "prompt_text": prompt_text,
                                "cached": meta.get("cached"),
                                "model_used": meta.get("model_used"),
                                "base_url": meta.get("base_url"),
                                "request_url": meta.get("request_url"),
                            }
                            if err:
                                row_log["error"] = err
                            _append_jsonl(prompt_log_path, [row_log])

                            if prompt_id == "10":
                                continue
                            if verdict == 0:
                                continue

                            if prompt_id == "04":
                                if verdict == 1:
                                    domain_arg, range_arg = arg1, arg2
                                elif verdict == 2:
                                    domain_arg, range_arg = arg2, arg1
                                else:
                                    continue
                            else:
                                domain_arg, range_arg = arg1, arg2

                            key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                            if key in seen_triples:
                                continue
                            seen_triples.add(key)
                            verified_rows.append(
                                {
                                    "id": sent_id,
                                    "sentence": sentence,
                                    "ontology_id": ontology_id,
                                    "relation_ja": rel_canon,
                                    "pid": pid,
                                    "prompt_id": prompt_id,
                                    "prompt_name": prompt_name,
                                    "domain_arg": domain_arg,
                                    "range_arg": range_arg,
                                    "domain_concept_ja": domain_concept,
                                    "range_concept_ja": range_concept,
                                    "verdict": verdict,
                                    "ast_uid": ast_uid,
                                    "stage": "verified",
                                }
                            )

                    else:
                        if len(x_values) < 2:
                            continue
                        if not domain_concept or not range_concept:
                            continue

                        def _judge_side(side: str, concept: str, argument: str, other_argument: str) -> int:
                            oa = other_argument if other_argument else pick_other_argument(x_values, argument)
                            call_key = (ontology_id, rel_canon, prompt_id, side, argument)
                            if call_key in seen_prompt_calls:
                                if log_dup_skips:
                                    _append_jsonl(
                                        prompt_log_path,
                                        [
                                            {
                                                "id": sent_id,
                                                "relation_ja": rel_canon,
                                                "relation_raw": rel_raw,
                                                "ontology_id": ontology_id,
                                                "prompt_id": prompt_id,
                                                "prompt_name": prompt_name,
                                                "mode": side,
                                                "argument": argument,
                                                "other_argument": oa,
                                                "verdict": None,
                                                "skipped_duplicate": True,
                                            }
                                        ],
                                    )
                                return 0
                            seen_prompt_calls.add(call_key)

                            prompt_text = render_prompt(
                                prompt,
                                {
                                    "relation_ja": rel_canon,
                                    "side": side,
                                    "concept_ja": concept,
                                    "argument": argument,
                                    "other_argument": oa,
                                    "context_sentence": sentence,
                                },
                            )
                            try:
                                v = ontology_judge.judge_prompt(prompt_text)
                            except Exception as e:
                                raise RuntimeError(
                                    f"ontology verify failed: id={sent_id!r} prompt_id={prompt_id!r} side={side!r} error={e}"
                                ) from e
                            meta = getattr(ontology_judge, "last_meta", {}) or {}
                            err = getattr(ontology_judge, "last_error", None)
                            row_log2: Dict[str, Any] = {
                                "id": sent_id,
                                "relation_ja": rel_canon,
                                "relation_raw": rel_raw,
                                "ontology_id": ontology_id,
                                "prompt_id": prompt_id,
                                "prompt_name": prompt_name,
                                "mode": side,
                                "argument": argument,
                                "other_argument": oa,
                                "verdict": v,
                                "prompt_text": prompt_text,
                                "cached": meta.get("cached"),
                                "model_used": meta.get("model_used"),
                                "base_url": meta.get("base_url"),
                                "request_url": meta.get("request_url"),
                            }
                            if err:
                                row_log2["error"] = err
                            _append_jsonl(prompt_log_path, [row_log2])
                            return int(v)

                        for a, b in _iter_allowed_pairs(x_values, parallel_value_groups):
                            vd = _judge_side("domain", domain_concept, a, b)
                            if vd == 1:
                                vr = _judge_side("range", range_concept, b, a)
                                if vr == 1:
                                    domain_arg, range_arg = a, b
                                    verdict = 1
                                    key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified_rows.append(
                                            {
                                                "id": sent_id,
                                                "sentence": sentence,
                                                "ontology_id": ontology_id,
                                                "relation_ja": rel_canon,
                                                "pid": pid,
                                                "prompt_id": prompt_id,
                                                "prompt_name": prompt_name,
                                                "domain_arg": domain_arg,
                                                "range_arg": range_arg,
                                                "domain_concept_ja": domain_concept,
                                                "range_concept_ja": range_concept,
                                                "verdict": verdict,
                                                "ast_uid": ast_uid,
                                                "stage": "verified",
                                            }
                                        )
                                    continue

                            vd = _judge_side("domain", domain_concept, b, a)
                            if vd == 1:
                                vr = _judge_side("range", range_concept, a, b)
                                if vr == 1:
                                    domain_arg, range_arg = b, a
                                    verdict = 1
                                    key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified_rows.append(
                                            {
                                                "id": sent_id,
                                                "sentence": sentence,
                                                "ontology_id": ontology_id,
                                                "relation_ja": rel_canon,
                                                "pid": pid,
                                                "prompt_id": prompt_id,
                                                "prompt_name": prompt_name,
                                                "domain_arg": domain_arg,
                                                "range_arg": range_arg,
                                                "domain_concept_ja": domain_concept,
                                                "range_concept_ja": range_concept,
                                                "verdict": verdict,
                                                "ast_uid": ast_uid,
                                                "stage": "verified",
                                            }
                                        )

            if verified_rows:
                for r in verified_rows:
                    wver.writerow(r)
                verified_rows_total += len(verified_rows)
                for r in verified_rows:
                    sub = strip_trailing_particles((r.get("domain_arg") or ""))
                    rel = (r.get("relation_ja") or "").strip()
                    obj = strip_trailing_particles((r.get("range_arg") or ""))
                    if not sub or not rel or not obj:
                        continue
                    key3 = (sub, rel, obj)
                    if key3 in seen_tr_set:
                        continue
                    seen_tr_set.add(key3)
                    triples_out.append({"sub": sub, "rel": rel, "obj": obj})

            _append_jsonl(extracted_jsonl_path, [{"id": sent_id, "sent_ja": sentence, "extracted_triples": triples_out}])

    print(f"verified rows: {verified_rows_total}")
    print(f"保存先(candidate): {candidate_csv}")
    print(f"保存先(verified): {verified_csv}")
    print(f"保存先(可視化): {vis_csv}")
    print(f"prompt log: {prompt_log_path}")
    print(f"extracted_triples: {extracted_jsonl_path}")
    print(f"logs: {run_log_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="select mode pipeline (default/no_verification/no_parallel_verification)")
    ap.add_argument(
        "--mode",
        default="default",
        choices=["default", "no_verification", "no_parallel_verification"],
        help="実行モード",
    )
    ap.add_argument("--pattern_index_json", default=PATTERN_INDEX_JSON_DEFAULT, help="patterns.index.json")
    ap.add_argument("--pattern_jsonl", default=PATTERN_JSONL_DEFAULT, help="patterns.jsonl")
    ap.add_argument("--input_jsonl_dir", default=INPUT_JSONL_DIR_DEFAULT, help="入力JSONLディレクトリ")
    ap.add_argument("--results_root", default=RESULTS_ROOT_DEFAULT, help="results root")
    ap.add_argument("--run_tag", default="auto", help="run tag (auto=timestamp)")
    ap.add_argument("--cache_mode", default="reuse", choices=["reuse", "refresh"], help="matchキャッシュ再利用/再生成")
    ap.add_argument("--export_candidate_slim", default="1", help="candidate_slim を出力するか (0/1)")
    ap.add_argument("--export_ast_repr", default=os.getenv("EXPORT_AST_REPR", ""), help="ast_repr 列を出すか (0/1)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.export_candidate_slim = str(args.export_candidate_slim).strip().lower() not in ("0", "false", "no", "")
    args.export_ast_repr = str(args.export_ast_repr).strip().lower() in ("1", "true", "yes", "on")
    if args.run_tag == "auto":
        args.run_tag = f"{_utc_tag()}__mode-{args.mode}"

    print("パターンJSONをロード中…")
    patterns = load_and_compile_patterns(index_path=args.pattern_index_json, jsonl_path=args.pattern_jsonl)
    ast_dict = build_ast_dict(patterns)

    # Pre-build matchers and parallel var groups once.
    total_matchers = 0
    for _v, entries in ast_dict.items():
        for e in entries:
            if "matcher" not in e:
                e["matcher"] = CKYMatcher(e["ast"], verbose=False)
            if "parallel_var_groups" not in e:
                e["parallel_var_groups"] = extract_parallel_variable_groups(e["ast"])
            total_matchers += 1
    print(f"ロード完了: {len(patterns)} パターン / CKYMatcher事前構築: {total_matchers}")

    patterns_fingerprint = compute_patterns_fingerprint(ast_dict)
    print(f"patterns_fingerprint: {patterns_fingerprint}")

    # Recurse under input_jsonl_dir to preserve directory structure per ontology/category.
    jsonl_paths = sorted(glob(os.path.join(args.input_jsonl_dir, "**", "*.jsonl"), recursive=True))
    jsonl_paths = [p for p in jsonl_paths if os.path.isfile(p)]
    if not jsonl_paths:
        print(f"入力JSONLが見つかりません: {args.input_jsonl_dir}")
        return

    for path in jsonl_paths:
        process_jsonl_select_mode(path, ast_dict, patterns_fingerprint, args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
