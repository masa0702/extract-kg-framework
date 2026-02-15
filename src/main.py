# extract_pred_arg_pair.py
# =============================================================
# ① AST Pickle をロード（ASTメタ付与: literal_list, parallel_var_count, ast_uid）
# ② GiNZA 依存解析 + CKY 表キャッシュを用意
# ③ メインプロセスが手動でパイプラインを監督
#     - GPUステージ（同時2本, 各文=子プロセス, device 0/1に割当）
#         → CKYAnalyzerで cky_dep を生成（タイムアウト超でプロセスkill）
#     - CPUステージ（多本, 各文=子プロセス）
#         → フィルタ→候補生成→CKYMatcher（合算タイムアウトでkill）
# ④ tqdm で進捗表示（結果は逐次CSV追記）
# ⑤ 診断ログ（sentence_stats / gpu_timing / gpu_done / gpu_timeout / cpu_timing / inflight）
# ⑥ 可視化ログ（どのASTがどの文に適用され、どの変数が何を拾ったか）をCSV追記
# =============================================================

# main.py（省略なし：長いのでそのまま全体を貼る方針に合わせています）
# ※あなたの手元コードと同名ファイルを置き換えてください

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import csv
import hashlib
import pickle
import pandas as pd
from itertools import product, combinations
from tqdm.auto import tqdm
from bisect import bisect_left, bisect_right
from glob import glob

import multiprocessing as mp
from multiprocessing.connection import Connection
import queue

import torch

from pattern.pattern_nodes import (
    ParallelNode,
    VariableNode,
)
from modules_core.matcher import CKYMatcher
from modules_core.cky_table import CkyTable
from modules_bert.bert_modules import CKYAnalyzer
from modules_core.bunsetu import DependencyAnalysis
from llm.parallel_judge import ParallelJudgeLLMJP
from config.filter_settings import PARALLEL_KEYS
from modules_core.pattern_compiler import load_and_compile_patterns, build_ast_dict
from modules_core.cache_store import SentenceCacheStore
from modules_core.ontology_verify import (
    get_ontology_resolver,
    get_ontology_judge,
    render_prompt,
    prompt_requires_pair,
    pick_other_argument,
    normalize_text,
    load_ontology_relation_aliases,
)
from modules_core.prompt_monitor import write_prompt_accept_summary
from modules_core.text_normalize import strip_trailing_particles

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PATTERN_INDEX_JSON = os.getenv(
    "PATTERN_INDEX_JSON",
    os.path.join(REPO_ROOT, "data/patterns/patterns.index.json"),
)
PATTERN_JSONL = os.getenv(
    "PATTERN_JSONL",
    os.path.join(REPO_ROOT, "data/patterns/patterns.jsonl"),
)
INPUT_JSONL_DIR = os.getenv(
    "INPUT_JSONL_DIR",
    # Prefer a dedicated test directory if present, otherwise default to the main target_data.
    (os.path.join(REPO_ROOT, "data/T2KGB_JA/test_target_data")
     if os.path.isdir(os.path.join(REPO_ROOT, "data/T2KGB_JA/test_target_data"))
     else os.path.join(REPO_ROOT, "data/T2KGB_JA/target_data")),
)
RESULTS_ROOT = os.getenv(
    "RESULTS_ROOT",
    os.path.join(REPO_ROOT, "results/extract_pred_arg_pair"),
)

EXPORT_AST_REPR = os.getenv("EXPORT_AST_REPR", "") == "1"

PROMPTS_JSON = os.getenv(
    "PROMPTS_JSON",
    os.path.join(REPO_ROOT, "prompts/prompts.json"),
)
RELATION_PROMPT_MAP_JSON = os.getenv(
    "RELATION_PROMPT_MAP_JSON",
    os.path.join(REPO_ROOT, "prompts/relation_prompt_map.json"),
)
ONTOLOGY_DIR = os.getenv(
    "ONTOLOGY_DIR",
    os.path.join(REPO_ROOT, "ontology"),
)

ONTOLOGY_ID_COL_CANDIDATES = [
    "ontology_id",
    "ontology",
    "ontology_category",
    "category",
]

EXCLUDE_POS = ["助詞", "接続詞", "助動詞",
               "補助記号-句点", "補助記号-読点",
               "記号-句点", "記号-読点"]

# Defaults for stability in iterative/debug runs.
# - Too many concurrent CPU workers can overload LLM-jp/vLLM and look like a "stall".
# - Periodic traceback dumps help identify where a stuck worker is blocking.
# Override any of these via environment variables.
DEFAULT_CPU_WORKERS = 4
DEFAULT_CPU_FAULTHANDLER_SEC = 60
DEFAULT_GPU_TIMEOUT_SEC = 1800
DEFAULT_GPU_FAULTHANDLER_SEC = 0

# Number of concurrent GPU-stage processes. When only 1 GPU is visible
# (e.g. CUDA_VISIBLE_DEVICES=0), setting this >1 is allowed but all workers
# will be mapped onto the available device(s) safely.
GPU_WORKERS            = max(1, int(os.getenv("GPU_WORKERS", "1")))
GPU_TIMEOUT_SEC        = int(os.getenv("GPU_TIMEOUT_SEC", str(DEFAULT_GPU_TIMEOUT_SEC)))
CPU_WORKERS            = int(os.getenv("CPU_WORKERS", str(DEFAULT_CPU_WORKERS)))
# Per-sentence CPU worker timeout. Huge values can hide hangs indefinitely; keep a reasonable default.
CPU_TOTAL_TIMEOUT_SEC  = int(os.getenv("CPU_TOTAL_TIMEOUT_SEC", "900"))

LIT_MAX_FREQ            = 20
CAND_SPAN_LIMIT_PER_AST = 1000

def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")

def preflight_ontology_llm() -> None:
    """
    オントロジー検証用 vLLM への到達性と、期待モデルの存在を 1 回だけ確認する。
    失敗時は原因を明示して停止（fail-fast）させる。
    """
    if not _env_flag("ONTOLOGY_VERIFY_PREFLIGHT", "1"):
        return
    # Import here to avoid import cycles in non-LLM usage.
    from llm.llmjp_client import get_llmjp_http_for

    client = get_llmjp_http_for("onto")
    expected = getattr(client, "model", None) or os.getenv("LLMJP_ONTO_MODEL") or os.getenv("LLMJP_MODEL") or "llmjp-13b"
    urls = list(getattr(client, "base_urls", None) or [])
    if not urls:
        urls = [str(getattr(client, "last_base_url", "") or "").strip()] if getattr(client, "last_base_url", None) else []
    urls = [u for u in urls if u]
    try:
        if not urls:
            raise RuntimeError("LLMJP_ONTO_BASE_URL(S) が空です")

        # Verify every onto base_url in the pool; otherwise a broken/mismatched server can slip through.
        for base in urls:
            data = client.list_models() if (len(urls) == 1) else None
            if data is None:
                # Force-request /models against this specific base_url (no round-robin).
                # We intentionally use the same Session/headers so this behaves like normal calls.
                url = f"{base.rstrip('/')}/models"
                client.last_base_url = base.rstrip("/")
                client.last_url = url
                r = client._session.get(url, headers=client._headers, timeout=client.timeout_sec)  # type: ignore[attr-defined]
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                r.raise_for_status()
                data = r.json()

            ids = set()
            for row in (data or {}).get("data", []) or []:
                mid = (row or {}).get("id")
                if mid:
                    ids.add(str(mid))
            if expected and ids and (expected not in ids):
                preview = sorted(ids)
                if len(preview) > 30:
                    preview = preview[:30] + ["...(truncated)"]
                raise RuntimeError(
                    "onto vLLM のモデル名が不整合です: "
                    f"expected={expected!r} url={base!r} available={preview!r}"
                )

        print(f"[preflight] ontology vLLM ok: model={expected!r} urls={urls!r}")
    except Exception as e:
        raise RuntimeError(
            "onto vLLM の preflight に失敗しました。"
            f" expected_model={expected!r} urls={urls!r} error={type(e).__name__}: {e}"
        ) from e

def extract_parallel_variables(ast):
    vars_ = []
    def visit(node):
        if isinstance(node, ParallelNode):
            if hasattr(node, "options"):
                for opt in node.options:
                    if isinstance(opt, VariableNode):
                        vars_.append(opt)
        for attr in ("elements", "options", "block"):
            if hasattr(node, attr):
                child = getattr(node, attr)
                if child:
                    if isinstance(child, list):
                        for c in child:
                            visit(c)
                    else:
                        visit(child)
    visit(ast)
    return [f"{v.symbol}{v.index}" for v in vars_]

def extract_parallel_variable_groups(ast):
    """Return list of variable-name groups for each ParallelNode (e.g. [['X1','X2'], ['Y1','Y2']])."""
    groups = []
    def visit(node):
        if isinstance(node, ParallelNode) and hasattr(node, "options"):
            g = []
            for opt in node.options or []:
                if isinstance(opt, VariableNode):
                    g.append(f"{opt.symbol}{opt.index}")
            if len(g) >= 2:
                groups.append(g)
        for attr in ("elements", "options", "block"):
            if hasattr(node, attr):
                child = getattr(node, attr)
                if child:
                    if isinstance(child, list):
                        for c in child:
                            visit(c)
                    else:
                        visit(child)
    visit(ast)
    return groups

def extract_relation_candidates_from_sentence(
    sentence: str,
    ontology_id: str,
    resolver,
) -> list[str]:
    if not sentence:
        return []
    ont = normalize_text(ontology_id)
    rows = getattr(resolver, "_rows", [])
    alias_pid = load_ontology_relation_aliases(ONTOLOGY_DIR, ont) if ont else {}
    pid_to_aliases = {}
    for a, pid in (alias_pid or {}).items():
        pid_to_aliases.setdefault(pid, []).append(a)
    out = []
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
    # preserve order of appearance in sentence
    uniq = []
    seen = set()
    for pred in out:
        if pred in seen:
            continue
        seen.add(pred)
        uniq.append(pred)
    return uniq

def clean_variable_mapping(varmap, clauses):
    new_map = {}
    for var, val in varmap.items():
        raw_val = str(val) if (val is not None) else ""
        found = None
        for cl in clauses:
            if cl[0] == val:
                found = cl
                break
            else:
                if val and isinstance(val, str) and cl[0] in val:
                    found = cl
                    break
        if not found:
            new_map[var] = strip_trailing_particles(raw_val)
            continue

        # Keep spaces inside bunsetsu (e.g., "New York") and keep internal particles ("太郎の車").
        surface = str(found[0]) if (found and len(found) > 0) else raw_val
        new_map[var] = strip_trailing_particles(surface, clause=found)
    return new_map

def build_sentence_text_and_offsets(clauses):
    surfaces = [str(cl[0]) for cl in clauses]
    starts = [0]; total = 0
    for s in surfaces:
        total += len(s); starts.append(total)
    return "".join(surfaces), starts

def find_all_occurrences(text, sub):
    pos_list = []; start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1: break
        pos_list.append(idx)
        start = idx + 1
    return pos_list

def count_occurrences_in_span(sorted_positions, span_start, span_end):
    lo = bisect_left(sorted_positions, span_start)
    hi = bisect_left(sorted_positions, span_end)
    return max(0, hi - lo)

def literals_in_order_within_span(literals, lit_pos_map, span_start, span_end):
    cur = span_start
    for lit in literals:
        pos_list = lit_pos_map.get(lit, [])
        i = bisect_left(pos_list, cur)
        found = False
        while i < len(pos_list):
            p = pos_list[i]
            if p >= span_end: break
            if p + len(lit) <= span_end:
                cur = p + len(lit); found = True; break
            i += 1
        if not found:
            return False
    return True

def get_ast_uid(ast) -> str:
    try:
        b = pickle.dumps(ast, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        b = repr(ast).encode("utf-8", errors="ignore")
    return hashlib.md5(b).hexdigest()[:16]

def iter_jsonl(path: str):
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

def gpu_child_worker(row_payload, device_id: int, conn: Connection):
    try:
        # Periodic traceback dumps for stuck GPU workers (opt-in via GPU_FAULTHANDLER_SEC).
        # Default is off to avoid noisy logs; enable when diagnosing GPU-stage stalls.
        try:
            import faulthandler
            sec = float(os.getenv("GPU_FAULTHANDLER_SEC", str(DEFAULT_GPU_FAULTHANDLER_SEC)) or 0.0)
            if sec and sec > 0:
                faulthandler.dump_traceback_later(sec, repeat=True)
        except Exception:
            pass

        # NOTE: CUDA_VISIBLE_DEVICES can make only a subset of GPUs visible.
        # If GPU_WORKERS > visible device count, naive set_device(device_id)
        # can raise "invalid device ordinal" and silently drop most rows.
        if torch.cuda.is_available():
            ndev = 0
            try:
                ndev = int(torch.cuda.device_count() or 0)
            except Exception:
                ndev = 0
            if ndev > 0:
                mapped = int(device_id) % ndev
                torch.cuda.set_device(mapped)
                device = f"cuda:{mapped}"
            else:
                device = "cpu"
        else:
            device = "cpu"

        analyzer = CKYAnalyzer()
        # Individual detectors handle device placement internally (mask_module/dep_bert).

        cky_table = row_payload["cky_table"]
        t0 = time.time()
        cky_dep = analyzer.analyze_cky_table(cky_table)
        t_analyze = time.time() - t0

        try:
            payload_size = len(pickle.dumps(cky_dep, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            payload_size = -1

        out = {
            "id": row_payload["id"],
            "sentence": row_payload["sent"],
            "ontology_id": row_payload.get("ontology_id", ""),
            "cky_dep": cky_dep,
            "clauses": row_payload["clauses"],
            "t_analyze": t_analyze,
            "payload_size": payload_size,
            "cky_stats": getattr(analyzer, "last_stats", None),
        }
        conn.send(out)
    except Exception as e:
        try:
            conn.send({"_error": str(e), "id": row_payload.get("id", "")})
        except Exception:
            pass
    finally:
        try:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def gpu_worker_loop(worker_id: int, device_id: int, in_q: mp.Queue, out_q: mp.Queue) -> None:
    """Persistent GPU-stage worker: load models once, then process many sentences.

    This fixes the biggest slowdown in the previous design (spawn-per-sentence),
    where mask-bert/dep-bert were loaded for every single sentence.
    """
    # NOTE: CUDA_VISIBLE_DEVICES can make only a subset of GPUs visible.
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
            try:
                payload_size = len(pickle.dumps(cky_dep, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                payload_size = -1
            out_q.put({
                "_worker_id": worker_id,
                "id": task.get("id", ""),
                "sentence": task.get("sent", ""),
                "ontology_id": task.get("ontology_id", ""),
                "cky_dep": cky_dep,
                "clauses": task.get("clauses", []),
                "t_analyze": t_analyze,
                "payload_size": payload_size,
                "cky_stats": getattr(analyzer, "last_stats", None),
            })
        except Exception as e:
            out_q.put({
                "_worker_id": worker_id,
                "id": task.get("id", ""),
                "_error": str(e),
            })

def cpu_child_worker(payload, ast_dict, conn: Connection):
    try:
        # ---- diagnostics (optional) ----
        # When CPU stage "stops", workers are often blocked inside matching or LLM calls.
        # These lightweight logs help pinpoint where time is spent per sentence.
        log_dir = payload.get("log_dir") or ""
        run_start_ts = float(payload.get("run_start_ts") or 0.0)
        sent_id_for_log = payload.get("id", "")

        def _log_stage(stage: str, **extra) -> None:
            if not log_dir:
                return
            try:
                path = os.path.join(log_dir, "cpu_stage.jsonl")
                ev = {
                    "ts_sec": (time.time() - run_start_ts) if run_start_ts else None,
                    "id": sent_id_for_log,
                    "stage": stage,
                }
                if extra:
                    ev.update(extra)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception:
                pass

        _log_stage("start")

        # Periodic traceback dumps for stuck workers (opt-in via CPU_FAULTHANDLER_SEC).
        try:
            import faulthandler
            sec = float(os.getenv("CPU_FAULTHANDLER_SEC", str(DEFAULT_CPU_FAULTHANDLER_SEC)) or 0.0)
            if (sec and sec > 0) and log_dir and sent_id_for_log:
                tb_path = os.path.join(log_dir, f"cpu_stall_{sent_id_for_log}.log")
                f = open(tb_path, "a", encoding="utf-8")
                faulthandler.enable(file=f)
                faulthandler.dump_traceback_later(sec, repeat=True, file=f)
        except Exception:
            pass

        judge = ParallelJudgeLLMJP()
        ontology_resolver = get_ontology_resolver(
            RELATION_PROMPT_MAP_JSON, PROMPTS_JSON, ONTOLOGY_DIR
        )
        ontology_judge = get_ontology_judge()
        ontology_id = payload.get("ontology_id") or ""
        sent_id  = payload["id"]; sentence = payload["sentence"]
        cky_dep  = payload["cky_dep"]; clauses = payload["clauses"]
        B = len(clauses)

        _log_stage("prepare", bunsetsu_cnt=B, sentence_len=len(sentence or ""))
        candidate_asts = []
        for v in range(2, B+1):
            if v in ast_dict:
                candidate_asts.extend(ast_dict[v])

        sent_text, starts = build_sentence_text_and_offsets(clauses)
        total_len = len(sent_text)
        full_start, full_end = 0, total_len

        par_starts = []
        for k in PARALLEL_KEYS:
            if k:
                ps = find_all_occurrences(sent_text, k)
                if ps: par_starts.extend(ps)
        par_starts.sort()
        parallel_sum_all = len(par_starts)

        uniq_literals = set()
        for e in candidate_asts:
            lits = e.get("literal_list", [])
            if lits: uniq_literals.update(lits)
        lit_pos_map = {lit: find_all_occurrences(sent_text, lit) for lit in uniq_literals}

        coarse_pass = []
        for entry in candidate_asts:
            literals  = entry.get("literal_list", [])
            par_cnt   = entry.get("parallel_var_count", 0)
            if par_cnt >= 2 and parallel_sum_all < (par_cnt - 1):
                continue
            if literals and not literals_in_order_within_span(literals, lit_pos_map, full_start, full_end):
                continue
            coarse_pass.append(entry)
        candidate_asts = coarse_pass

        t0 = time.time()
        filtered = []

        _log_stage("span_filter_start", cand_asts=len(candidate_asts))
        for entry in candidate_asts:
            var_count = entry.get("var_count", 0)
            literals  = entry.get("literal_list", [])
            par_cnt   = entry.get("parallel_var_count", 0)

            eff_lits = []
            if literals:
                for lit in literals:
                    freq = len(lit_pos_map.get(lit, []))
                    if freq == 0:
                        eff_lits = []; break
                    if freq <= LIT_MAX_FREQ:
                        eff_lits.append(lit)
                if not eff_lits:
                    eff_lits = [max(literals, key=len)]

            # 候補 (i,j)（まずは素のspanだけ。拡張は passed 後に行う）
            cand_ij_base = []
            if eff_lits:
                first = eff_lits[0]
                first_pos = lit_pos_map.get(first, [])
                for p0 in first_pos:
                    cur_end = p0 + len(first)
                    ok = True
                    for lit in eff_lits[1:]:
                        lst = lit_pos_map.get(lit, [])
                        idx = bisect_left(lst, cur_end)
                        if idx >= len(lst): ok = False; break
                        pos = lst[idx]; cur_end = pos + len(lit)
                        if cur_end > total_len: ok = False; break
                    if not ok: continue
                    span_start, span_end = p0, cur_end
                    i = max(0, bisect_right(starts, span_start) - 1)
                    j = max(0, bisect_left(starts, span_end) - 1)
                    if j <= i: j = min(B - 1, i + 1)
                    cand_ij_base.append((i, j))
            else:
                cand_ij_base = [(0, B - 1)]

            passed = False
            seen_ij = set()
            for (i, j) in cand_ij_base:
                if (i, j) in seen_ij: continue
                seen_ij.add((i, j))
                chunk_num = j - i + 1
                if var_count > chunk_num: continue
                if par_cnt >= 2:
                    span_start = starts[i]; span_end = starts[j + 1]
                    cnt = count_occurrences_in_span(par_starts, span_start, span_end)
                    if cnt < par_cnt - 1: continue
                passed = True; break

            if passed:
                # passed したものだけ span 拡張を追加（走査セル制限用）
                cand_spans = set(cand_ij_base)
                for (i, j) in list(cand_spans):
                    if i > 0:
                        cand_spans.add((i - 1, j))
                    if j < B - 1:
                        cand_spans.add((i, j + 1))

                # CKYMatcher は 1-based なので変換して保持
                entry["cand_spans"] = sorted(
                    [(i + 1, j + 1) for (i, j) in cand_spans if 0 <= i <= j < B],
                    key=lambda x: (-(x[1] - x[0]), x[0], x[1]),
                )
                filtered.append(entry)

        t_filter = time.time() - t0

        _log_stage("span_filter_done", filtered_asts=len(filtered), t_filter_sec=round(t_filter, 6))
        t1 = time.time()
        seen = set()
        candidates = []
        vis_rows = []
        verified = []
        seen_triples = set()
        prompt_logs = []
        seen_prompt_calls = set()
        log_dup_skips = str(os.getenv("LOG_DUPLICATE_SKIPS", "")).lower() in ("1", "true", "yes")
        # Per-sentence cache: raw relation string -> resolved row / canonical label
        rel_row_cache: dict[str, dict] = {}
        rel_label_cache: dict[str, str] = {}

        def _is_parallel_pair(a: str, b: str, parallel_value_groups: list[list[str]]) -> bool:
            if not a or not b:
                return False
            for g in parallel_value_groups:
                s = set(g)
                if a in s and b in s:
                    return True
            return False

        def _iter_allowed_pairs(x_values: list[str], parallel_value_groups: list[list[str]]):
            # Use unordered pairs; skip pairs that are within the same verified-parallel group.
            for a1, a2 in combinations(x_values, 2):
                if _is_parallel_pair(a1, a2, parallel_value_groups):
                    continue
                yield a1, a2

        _log_stage("matching_start")
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

                varmap_raw   = dict(r.variable_mapping)
                varmap_clean = clean_variable_mapping(varmap_raw, clauses)

                # For logging/debug (not used for filtering).
                par_names = extract_parallel_variables(ast)
                par_elems = [varmap_clean[name] for name in par_names if name in varmap_clean]

                # Parallel validation per ParallelNode group (avoid mixing independent groups).
                parallel_value_groups: list[list[str]] = []
                if par_var_groups:
                    ok_parallel = True
                    for g in par_var_groups:
                        vals = [varmap_clean.get(name, "") for name in g]
                        vals = [v for v in vals if v]
                        if len(vals) < 2:
                            continue
                        if judge.judge_parallel(sentence, vals) is False:
                            ok_parallel = False
                            break
                        parallel_value_groups.append(vals)
                    if not ok_parallel:
                        continue

                Xs = []; Ys = []
                for k, v2 in varmap_clean.items():
                    if k.startswith("X"): Xs.append((k, v2))
                    elif k.startswith("Y"): Ys.append((k, v2))
                Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
                Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())
                if not Xs or not Ys:
                    continue

                ast_uid  = entry.get("ast_uid", get_ast_uid(ast))
                for _, ((xk, xv), (yk, yv)) in enumerate(product(Xs, Ys)):
                    rel_out = yv
                    if yv in rel_label_cache:
                        rel_out = rel_label_cache[yv]
                    else:
                        row = ontology_resolver.resolve_relation_row(yv, ontology_id)
                        if row:
                            rel_row_cache[yv] = row
                            rel_out = (
                                ontology_resolver.canonical_relation_label(row, ontology_id)
                                or normalize_text(row.get("predicate_ja"))
                                or yv
                            )
                        rel_label_cache[yv] = rel_out
                    candidates.append({
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
                    })

                var_cnt  = entry.get("var_count", 0)
                par_cnt  = entry.get("parallel_var_count", 0)
                literals = entry.get("literal_list", [])

                # Ontology verification (pattern-level candidates)
                x_values = [v for _, v in Xs]
                y_values = [v for _, v in Ys]

                resolved_rows = {}
                for rel in y_values:
                    row = rel_row_cache.get(rel) or ontology_resolver.resolve_relation_row(rel, ontology_id)
                    if row:
                        rel_row_cache[rel] = row
                        resolved_rows[rel] = row

                if not resolved_rows:
                    fallback_rels = extract_relation_candidates_from_sentence(
                        sentence,
                        ontology_id,
                        ontology_resolver,
                    )
                    for rel in fallback_rels:
                        row = rel_row_cache.get(rel) or ontology_resolver.resolve_relation_row(rel, ontology_id)
                        if row:
                            rel_row_cache[rel] = row
                            resolved_rows[rel] = row

                for rel, row in resolved_rows.items():
                    rel_raw = rel
                    rel_canon = (
                        ontology_resolver.canonical_relation_label(row, ontology_id)
                        or normalize_text(row.get("predicate_ja"))
                        or rel_raw
                    )
                    prompt_id = row.get("prompt_id", "")
                    prompt = ontology_resolver.get_prompt(prompt_id)
                    if not prompt:
                        continue

                    domain_concept, range_concept = ontology_resolver.resolve_concepts(row, ontology_id)
                    domain_concept = domain_concept or ""
                    range_concept = range_concept or ""

                    pid = row.get("pid", "")
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
                                    prompt_logs.append({
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
                                    })
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
                                verdict = ontology_judge.judge_prompt(prompt_text)
                                _m = getattr(ontology_judge, "last_meta", {}) or {}
                                _e = getattr(ontology_judge, "last_error", None)
                                _row = {
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
                                    "cached": _m.get("cached"),
                                    "model_used": _m.get("model_used"),
                                    "temperature": _m.get("temperature"),
                                    "max_tokens": _m.get("max_tokens"),
                                    "base_url": _m.get("base_url"),
                                    "request_url": _m.get("request_url"),
                                }
                            if _e:
                                _row["error"] = _e
                            prompt_logs.append(_row)

                            # prompt_id=10 is a synonym/normalization gate (not a domain/range verifier).
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
                                # Unknown pair-prompt: accept only in the given order.
                                domain_arg, range_arg = arg1, arg2

                            key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                            if key in seen_triples:
                                continue
                            seen_triples.add(key)
                            verified.append({
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
                            })
                    else:
                        # Binary side-based verifier (01/15/17/21):
                        # Decide a (domain, range) pair; do not accept single-sided matches.
                        if len(x_values) < 2:
                            continue
                        if not domain_concept or not range_concept:
                            continue
                        prompt_side_verdict_cache: Dict[Tuple[str, str, str, str, str], int] = {}

                        def _judge_side(side: str, concept: str, argument: str, other_argument: str) -> int:
                                oa = other_argument if other_argument else pick_other_argument(x_values, argument)
                                call_key = (ontology_id, rel_canon, prompt_id, side, argument)
                                if call_key in prompt_side_verdict_cache:
                                    if log_dup_skips:
                                        prompt_logs.append(
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
                                                "verdict": prompt_side_verdict_cache.get(call_key),
                                                "skipped_duplicate": True,
                                            }
                                        )
                                    return int(prompt_side_verdict_cache.get(call_key, 0))

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
                                    t_override = None
                                    mt_override = None
                                    if str(prompt_id) == "21":
                                        try:
                                            t_override = float(os.getenv("PROMPT21_TEMPERATURE", "0.15"))
                                        except Exception:
                                            t_override = 0.15
                                        try:
                                            s_mt = os.getenv("PROMPT21_MAX_TOKENS", "").strip()
                                            mt_override = int(s_mt) if s_mt else None
                                        except Exception:
                                            mt_override = None
                                    v = ontology_judge.judge_prompt(prompt_text, temperature=t_override, max_tokens=mt_override)
                                except Exception as e:
                                    raise RuntimeError(
                                        f"ontology verify failed: id={sent_id!r} prompt_id={prompt_id!r} side={side!r} error={e}"
                                    ) from e
                                meta = getattr(ontology_judge, "last_meta", {}) or {}
                                err = getattr(ontology_judge, "last_error", None)

                                final_v = int(v)
                                fallback_used = False
                                fallback_verdict = None
                                if str(prompt_id) == "21" and final_v == 0:
                                    prompt_fb = ontology_resolver.get_prompt("22")
                                    if prompt_fb:
                                        prompt_text_fb = render_prompt(
                                            prompt_fb,
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
                                            v_fb = ontology_judge.judge_prompt(prompt_text_fb, temperature=0.0)
                                        except Exception as e:
                                            raise RuntimeError(
                                                f"ontology verify failed (fallback): id={sent_id!r} prompt_id='22' side={side!r} error={e}"
                                            ) from e
                                        meta_fb = getattr(ontology_judge, "last_meta", {}) or {}
                                        err_fb = getattr(ontology_judge, "last_error", None)
                                        row_log_fb: Dict[str, Any] = {
                                            "id": sent_id,
                                            "relation_ja": rel_canon,
                                            "relation_raw": rel_raw,
                                            "ontology_id": ontology_id,
                                            "prompt_id": "22",
                                            "prompt_name": getattr(prompt_fb, "prompt_name", ""),
                                            "mode": side,
                                            "argument": argument,
                                            "other_argument": oa,
                                            "verdict": v_fb,
                                            "prompt_text": prompt_text_fb,
                                            "cached": meta_fb.get("cached"),
                                            "model_used": meta_fb.get("model_used"),
                                            "temperature": meta_fb.get("temperature"),
                                            "max_tokens": meta_fb.get("max_tokens"),
                                            "base_url": meta_fb.get("base_url"),
                                            "request_url": meta_fb.get("request_url"),
                                            "fallback_from": "21",
                                            "fallback_to": "22",
                                            "fallback_used": True,
                                            "primary_verdict": final_v,
                                        }
                                        if err_fb:
                                            row_log_fb["error"] = err_fb
                                        prompt_logs.append(row_log_fb)
                                        fallback_used = True
                                        fallback_verdict = int(v_fb)
                                        if int(v_fb) == 1:
                                            final_v = 1

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
                                    "temperature": meta.get("temperature"),
                                    "max_tokens": meta.get("max_tokens"),
                                    "base_url": meta.get("base_url"),
                                    "request_url": meta.get("request_url"),
                                    "fallback_to": "22" if str(prompt_id) == "21" else None,
                                    "fallback_used": fallback_used,
                                    "fallback_verdict": fallback_verdict,
                                    "final_verdict": final_v,
                                }
                                if err:
                                    row_log2["error"] = err
                                prompt_logs.append(row_log2)
                                prompt_side_verdict_cache[call_key] = int(final_v)
                                return int(final_v)

                        for a, b in _iter_allowed_pairs(x_values, parallel_value_groups):
                            # Try (domain=a, range=b)
                            vd = _judge_side("domain", domain_concept, a, b)
                            if vd == 1:
                                vr = _judge_side("range", range_concept, b, a)
                                if vr == 1:
                                    domain_arg, range_arg = a, b
                                    verdict = 1
                                    key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified.append({
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
                                        })
                                    continue

                            # Try swapped (domain=b, range=a)
                            vd = _judge_side("domain", domain_concept, b, a)
                            if vd == 1:
                                vr = _judge_side("range", range_concept, a, b)
                                if vr == 1:
                                    domain_arg, range_arg = b, a
                                    verdict = 1
                                    key = (sent_id, rel_canon, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified.append({
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
                                        })

                vis_row = {
                    "id": sent_id,
                    "sentence": sentence,
                    "ast_uid": ast_uid,
                    "pattern_id": entry.get("pattern_id", ""),
                    "pattern": entry.get("pattern", ""),
                    "var_count": var_cnt,
                    "parallel_var_count": par_cnt,
                    "literals": "|".join(literals) if literals else "",
                    "X_vars": "|".join([f"{k}:{v}" for k,v in Xs]) if Xs else "",
                    "Y_vars": "|".join([f"{k}:{v}" for k,v in Ys]) if Ys else "",
                    "varmap_raw": json.dumps(varmap_raw, ensure_ascii=False),
                    "varmap_clean": json.dumps(varmap_clean, ensure_ascii=False),
                    "parallel_var_names": json.dumps(par_names, ensure_ascii=False),
                    "parallel_elements": json.dumps(par_elems, ensure_ascii=False),
                }
                if EXPORT_AST_REPR:
                    vis_row["ast_repr"] = repr(ast)
                vis_rows.append(vis_row)

        t_match = time.time() - t1
        _log_stage("matching_done", t_match_sec=round(t_match, 6), cand_rows=len(candidates), verified_rows=len(verified))

        # Write large results to shard files and send only small metadata via Pipe.
        # This avoids a deadlock where the child blocks in conn.send() while the parent
        # waits for the process to exit before reading.
        shard_dir = os.path.join(payload.get("log_dir") or "", "cpu_shards")
        os.makedirs(shard_dir, exist_ok=True)
        safe_id = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(sent_id))
        shard_base = f"{safe_id}.{os.getpid()}"
        cand_shard = os.path.join(shard_dir, f"{shard_base}.candidates.jsonl")
        vis_shard = os.path.join(shard_dir, f"{shard_base}.vis.jsonl")
        verified_shard = os.path.join(shard_dir, f"{shard_base}.verified.jsonl")
        prompt_shard = os.path.join(shard_dir, f"{shard_base}.prompt.jsonl")

        def _write_jsonl(path: str, rows: list[dict]) -> None:
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _write_jsonl(cand_shard, candidates)
        _write_jsonl(vis_shard, vis_rows)
        _write_jsonl(verified_shard, verified)
        _write_jsonl(prompt_shard, prompt_logs)

        out = {
            "id": sent_id,
            "sentence": sentence,
            "ontology_id": ontology_id,
            "shards": {
                "candidate_jsonl": cand_shard,
                "vis_jsonl": vis_shard,
                "verified_jsonl": verified_shard,
                "prompt_jsonl": prompt_shard,
            },
            "rows": {
                "candidate": len(candidates),
                "vis": len(vis_rows),
                "verified": len(verified),
                "prompt": len(prompt_logs),
            },
            "t_filter": t_filter,
            "t_match": t_match,
            "cand_asts": len(candidate_asts),
            "filtered_asts": len(filtered)
        }
        _log_stage("done")
        conn.send(out)
    except Exception as e:
        try:
            conn.send({"_error": str(e), "id": payload.get("id", "")})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

# =============================================================
# main
# =============================================================
def process_jsonl(input_jsonl_path: str, ast_dict: dict) -> None:
    dir_name = os.path.basename(os.path.dirname(input_jsonl_path))
    filename = os.path.basename(input_jsonl_path)
    prefix = filename[:-6] if filename.endswith(".jsonl") else os.path.splitext(filename)[0]
    output_dir = os.path.join(RESULTS_ROOT, dir_name, prefix)
    log_dir = os.path.join(output_dir, "logs")
    cache_dir = os.path.join(output_dir, "cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"入力JSONL: {input_jsonl_path}")

    records = list(iter_jsonl(input_jsonl_path))
    if not records:
        print("JSONLが空です。スキップします。")
        return

    sentences = []
    seen_sentences = set()
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
        print("GiNZA 解析: {} 文".format(len(new_sentences)))
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

    rows = []
    for r in records:
        s = r.get("sent_ja") or r.get("sent") or ""
        if not s or s not in cky_data:
            continue
        info = cky_data[s]
        ontology_id = ""
        for col in ONTOLOGY_ID_COL_CANDIDATES:
            val = r.get(col)
            if val is not None and str(val).strip() != "":
                ontology_id = str(val)
                break
        if not ontology_id:
            ontology_id = default_ontology_id or file_ontology_id
        rows.append({
            "id": r.get("id", ""),
            "sent": s,
            "cky_table": info.get("dependency_table", []),
            "clauses": info.get("clauses", []),
            "ontology_id": ontology_id,
        })

    total_gpu = len(rows)
    total_cpu = total_gpu
    if total_gpu == 0:
        print("CKY情報のある文がありません。スキップします。")
        return

    candidate_csv = os.path.join(output_dir, f"{prefix}_triples_candidate.csv")
    candidate_slim_csv = os.path.join(output_dir, f"{prefix}_triples_candidate_slim.csv")
    verified_csv = os.path.join(output_dir, f"{prefix}_triples_verified.csv")
    vis_csv = os.path.join(output_dir, f"{prefix}_ast_visualization.csv")

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
    if not os.path.exists(candidate_csv):
        pd.DataFrame(columns=triple_cols).to_csv(
            candidate_csv, index=False, encoding="utf-8-sig"
        )
    slim_cols = ["id", "sentence", "ontology_id", "relation_ja", "domain_arg", "range_arg", "ast_uid", "stage"]
    if str(os.getenv("EXPORT_CANDIDATE_SLIM", "1")).strip().lower() not in ("0", "false", "no", ""):
        if not os.path.exists(candidate_slim_csv):
            pd.DataFrame(columns=slim_cols).to_csv(
                candidate_slim_csv, index=False, encoding="utf-8-sig"
            )
    if not os.path.exists(verified_csv):
        pd.DataFrame(columns=triple_cols).to_csv(
            verified_csv, index=False, encoding="utf-8-sig"
        )

    vis_cols = [
        "id","sentence","ast_uid","pattern_id","pattern","var_count","parallel_var_count","literals",
        "X_vars","Y_vars","varmap_raw","varmap_clean","parallel_var_names","parallel_elements"
    ]
    if EXPORT_AST_REPR:
        vis_cols.append("ast_repr")
    if not os.path.exists(vis_csv):
        pd.DataFrame(columns=vis_cols).to_csv(
            vis_csv, index=False, encoding="utf-8-sig"
        )

    def _append_csv_rows(path: str, cols: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        # IMPORTANT: Append with utf-8 (no BOM). The header file is created with utf-8-sig once.
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    def _append_csv_from_jsonl(path_csv: str, cols: list[str], path_jsonl: str) -> None:
        if not path_jsonl or (not os.path.exists(path_jsonl)):
            return
        with open(path_csv, "a", newline="", encoding="utf-8") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=cols, extrasaction="ignore")
            with open(path_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    w.writerow({c: r.get(c, "") for c in cols})

    sent_stats_csv = os.path.join(log_dir, f"{prefix}_sentence_stats.csv")
    gpu_timing_csv = os.path.join(log_dir, f"{prefix}_gpu_timing.csv")
    gpu_done_csv = os.path.join(log_dir, f"{prefix}_gpu_done.csv")
    gpu_timeout_csv = os.path.join(log_dir, f"{prefix}_gpu_timeout.csv")
    gpu_error_jsonl = os.path.join(log_dir, f"{prefix}_gpu_errors.jsonl")
    gpu_started_csv = os.path.join(log_dir, f"{prefix}_gpu_started.csv")
    gpu_stats_jsonl = os.path.join(log_dir, f"{prefix}_gpu_stats.jsonl")
    cpu_timing_csv = os.path.join(log_dir, f"{prefix}_cpu_timing.csv")
    cpu_error_jsonl = os.path.join(log_dir, f"{prefix}_cpu_errors.jsonl")
    inflight_csv = os.path.join(log_dir, f"{prefix}_inflight.csv")
    prompt_log_path = os.path.join(output_dir, f"{prefix}_prompt_log.jsonl")
    if not os.path.exists(prompt_log_path):
        with open(prompt_log_path, "w", encoding="utf-8") as _:
            pass
    extracted_jsonl_path = os.path.join(output_dir, f"{prefix}_extracted_triples.jsonl")
    if not os.path.exists(extracted_jsonl_path):
        with open(extracted_jsonl_path, "w", encoding="utf-8") as _:
            pass

    if not os.path.exists(sent_stats_csv):
        with open(sent_stats_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","sentence_len","bunsetsu_cnt","cells","parallel_sum_all","cand_ast_estimate"])
    if not os.path.exists(gpu_timing_csv):
        with open(gpu_timing_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","t_analyze_sec","payload_size_bytes"])
    if not os.path.exists(gpu_done_csv):
        with open(gpu_done_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","id"])
    if not os.path.exists(gpu_timeout_csv):
        with open(gpu_timeout_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","id"])
    if not os.path.exists(gpu_error_jsonl):
        with open(gpu_error_jsonl, "w", encoding="utf-8") as _:
            pass
    if not os.path.exists(gpu_started_csv):
        with open(gpu_started_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","id","bunsetsu_cnt","cells","sentence_len"])
    if not os.path.exists(gpu_stats_jsonl):
        with open(gpu_stats_jsonl, "w", encoding="utf-8") as _:
            pass
    if not os.path.exists(cpu_timing_csv):
        with open(cpu_timing_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","t_filter_sec","t_match_sec","cand_asts","filtered_asts","timeout"])
    if not os.path.exists(cpu_error_jsonl):
        with open(cpu_error_jsonl, "w", encoding="utf-8") as _:
            pass
    if not os.path.exists(inflight_csv):
        with open(inflight_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","inflight_gpu","inflight_cpu","done_gpu","done_cpu","submitted_gpu","submitted_cpu"])

    with open(sent_stats_csv, "a", newline="", encoding="utf-8") as fstats:
        wstats = csv.writer(fstats)
        for row in rows:
            s = row["sent"]
            clauses = row["clauses"]
            B = len(clauses)
            cells = B * (B - 1) // 2
            sent_text, _starts = build_sentence_text_and_offsets(clauses)
            par_total = 0
            for k in PARALLEL_KEYS:
                if k:
                    par_total += len(find_all_occurrences(sent_text, k))
            cand_est = 0
            for v in range(2, B + 1):
                cand_est += len(ast_dict.get(v, []))
            wstats.writerow([row["id"], len(s), B, cells, par_total, cand_est])

    global_ast_dict = ast_dict

    start_ts = time.time()

    ctx_gpu = mp.get_context("spawn")
    try:
        ctx_cpu = mp.get_context("fork")
    except ValueError:
        ctx_cpu = mp.get_context("spawn")

    # ----------------------------
    # GPU stage: persistent workers
    # ----------------------------
    # Previous design (1 sentence = 1 spawned process) re-loaded mask-bert/dep-bert for every sentence,
    # making the GPU stage extremely slow and prone to appearing "stuck".
    # We keep a small number of persistent GPU workers and feed them tasks.
    gpu_out_q: mp.Queue = ctx_gpu.Queue()
    gpu_workers: list[dict] = []
    for wid in range(GPU_WORKERS):
        inq = ctx_gpu.Queue(maxsize=1)
        p = ctx_gpu.Process(target=gpu_worker_loop, args=(wid, wid, inq, gpu_out_q), daemon=True)
        p.start()
        gpu_workers.append({
            "wid": wid,
            "proc": p,
            "inq": inq,
            "busy": False,
            "start": 0.0,
            "row_id": "",
            "row_payload": None,
            "retries": 0,
        })
    GPU_MAX_RETRIES_PER_ROW = 1

    cpu_slots = []
    gpu_idx = 0
    done_gpu = 0
    submitted_gpu = 0
    submitted_cpu = 0
    done_cpu = 0

    rows.sort(key=lambda r: (len(r["clauses"]) * (len(r["clauses"])-1)) // 2)

    gpu_pbar = tqdm(total=total_gpu, desc="GPU stage")
    cpu_pbar = tqdm(total=total_cpu, desc="CPU stage")

    cpu_queue = []

    last_inflight_log = time.time()
    def log_inflight():
        nonlocal last_inflight_log
        now = time.time()
        if now - last_inflight_log >= 5.0:
            os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(inflight_csv):
                with open(inflight_csv, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["ts_sec","inflight_gpu","inflight_cpu","done_gpu","done_cpu","submitted_gpu","submitted_cpu"])
            inflight_gpu = submitted_gpu - done_gpu
            inflight_cpu = submitted_cpu - done_cpu
            with open(inflight_csv, "a", newline="", encoding="utf-8") as finf:
                csv.writer(finf).writerow([f'{now - start_ts:.3f}', inflight_gpu, inflight_cpu,
                                           done_gpu, done_cpu, submitted_gpu, submitted_cpu])
            last_inflight_log = now

    def _any_gpu_busy() -> bool:
        return any(w.get("busy") for w in gpu_workers)

    gpu_workers_stopped = False
    abort_error = None

    while (done_gpu < total_gpu) or _any_gpu_busy() or cpu_queue or cpu_slots:
        # Submit tasks to idle GPU workers.
        for w in gpu_workers:
            if gpu_idx >= total_gpu:
                break
            if w["busy"]:
                continue
            row = rows[gpu_idx]
            row_payload = dict(row)
            row_payload["log_dir"] = log_dir
            row_payload["run_start_ts"] = start_ts

            try:
                B = len(row.get("clauses", []) or [])
                cells = B * (B - 1) // 2
                with open(gpu_started_csv, "a", newline="", encoding="utf-8") as fgs:
                    csv.writer(fgs).writerow([f'{time.time()-start_ts:.3f}', row.get("id",""), B, cells, len(row.get("sent","") or "")])
            except Exception:
                pass

            w["inq"].put(row_payload)
            w["busy"] = True
            w["start"] = time.time()
            w["row_id"] = row.get("id", "")
            w["row_payload"] = row_payload
            w["retries"] = 0
            submitted_gpu += 1
            gpu_idx += 1

        # Collect completed GPU results.
        while True:
            try:
                payload = gpu_out_q.get_nowait()
            except queue.Empty:
                break

            wid = int(payload.get("_worker_id", -1))
            w = gpu_workers[wid] if (0 <= wid < len(gpu_workers)) else None
            row_id = payload.get("id") or (w.get("row_id") if w else "")

            if w:
                w["busy"] = False
                w["row_id"] = ""
                w["row_payload"] = None
                w["retries"] = 0

            with open(gpu_done_csv, "a", newline="", encoding="utf-8") as fgd:
                csv.writer(fgd).writerow([f'{time.time()-start_ts:.3f}', row_id])

            if payload and "_error" in payload:
                try:
                    with open(gpu_error_jsonl, "a", encoding="utf-8") as ferr:
                        ferr.write(json.dumps({
                            "ts_sec": time.time() - start_ts,
                            "id": row_id,
                            "error": payload.get("_error", ""),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                done_gpu += 1
                gpu_pbar.update(1)
                continue

            if payload:
                with open(gpu_timing_csv, "a", newline="", encoding="utf-8") as fgpu:
                    csv.writer(fgpu).writerow([payload.get("id",""),
                                               f'{payload.get("t_analyze", 0.0):.6f}',
                                               payload.get("payload_size", -1)])
                try:
                    with open(gpu_stats_jsonl, "a", encoding="utf-8") as fgs:
                        fgs.write(json.dumps({
                            "ts_sec": time.time() - start_ts,
                            "id": payload.get("id", row_id),
                            "t_analyze_sec": payload.get("t_analyze", 0.0),
                            "stats": payload.get("cky_stats"),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass

                payload["log_dir"] = log_dir
                payload["run_start_ts"] = start_ts
                cpu_queue.append(payload)

            done_gpu += 1
            gpu_pbar.update(1)

        # GPU worker timeouts / crashed workers.
        for w in gpu_workers:
            if not w["busy"]:
                # Restart if crashed unexpectedly.
                if not w["proc"].is_alive():
                    try:
                        w["proc"].join(timeout=0.1)
                    except Exception:
                        pass
                    inq = ctx_gpu.Queue(maxsize=1)
                    p = ctx_gpu.Process(target=gpu_worker_loop, args=(w["wid"], w["wid"], inq, gpu_out_q), daemon=True)
                    p.start()
                    w["inq"] = inq
                    w["proc"] = p
                continue

            if not w["proc"].is_alive():
                # Busy but dead: treat as error and continue.
                row_id = w.get("row_id", "")
                try:
                    with open(gpu_error_jsonl, "a", encoding="utf-8") as ferr:
                        ferr.write(json.dumps({
                            "ts_sec": time.time() - start_ts,
                            "id": row_id,
                            "error": "gpu_worker_died",
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                w["busy"] = False
                w["row_id"] = ""
                w["row_payload"] = None
                done_gpu += 1
                gpu_pbar.update(1)
                continue

            if (time.time() - w["start"]) > GPU_TIMEOUT_SEC:
                row_id = w.get("row_id", "")
                try:
                    w["proc"].terminate()
                except Exception:
                    pass
                try:
                    w["proc"].join(timeout=0.5)
                except Exception:
                    pass

                try:
                    with open(gpu_timeout_csv, "a", newline="", encoding="utf-8") as fgt:
                        csv.writer(fgt).writerow([f'{time.time()-start_ts:.3f}', row_id])
                except Exception:
                    pass

                # Restart the worker process (models are reloaded in the new process).
                inq = ctx_gpu.Queue(maxsize=1)
                p = ctx_gpu.Process(target=gpu_worker_loop, args=(w["wid"], w["wid"], inq, gpu_out_q), daemon=True)
                p.start()
                w["inq"] = inq
                w["proc"] = p

                # Retry once on a fresh process; otherwise mark as done (timeout).
                retry_payload = w.get("row_payload")
                if (retry_payload is not None) and (w.get("retries", 0) < GPU_MAX_RETRIES_PER_ROW):
                    w["retries"] = w.get("retries", 0) + 1
                    w["start"] = time.time()
                    w["inq"].put(retry_payload)
                else:
                    w["busy"] = False
                    w["row_id"] = ""
                    w["row_payload"] = None
                    w["retries"] = 0
                    done_gpu += 1
                    gpu_pbar.update(1)

        # If GPU stage is fully done, stop GPU workers early to release GPU memory for llm-jp/vLLM.
        # This can materially speed up the CPU stage on 1-GPU machines.
        if (not gpu_workers_stopped) and (done_gpu >= total_gpu) and (not _any_gpu_busy()):
            for w in gpu_workers:
                try:
                    w["inq"].put(None)
                except Exception:
                    pass
            for w in gpu_workers:
                try:
                    w["proc"].join(timeout=2.0)
                except Exception:
                    pass
                if w["proc"].is_alive():
                    try:
                        w["proc"].terminate()
                    except Exception:
                        pass
            gpu_workers_stopped = True

        while len(cpu_slots) < CPU_WORKERS and cpu_queue:
            payload = cpu_queue.pop(0)
            parent_conn, child_conn = ctx_cpu.Pipe(duplex=False)
            p = ctx_cpu.Process(target=cpu_child_worker, args=(payload, global_ast_dict, child_conn), daemon=True)
            p.start()
            submitted_cpu += 1
            cpu_slots.append({
                "proc": p,
                "conn": parent_conn,
                "start": time.time(),
                "row_id": payload["id"],
                "sentence": payload["sentence"],
            })

        still_cpu = []
        for slot in cpu_slots:
            p: mp.Process = slot["proc"]
            conn: Connection = slot["conn"]
            row_id = slot["row_id"]
            started = slot["start"]

            if not p.is_alive():
                out = None
                try:
                    for _ in range(3):
                        if conn.poll(0.05):
                            out = conn.recv()
                            break
                        time.sleep(0.005)
                except EOFError:
                    out = None
                finally:
                    try: conn.close()
                    except Exception: pass
                    p.join(timeout=0.1)

                if out and "_error" in out:
                    try:
                        with open(cpu_error_jsonl, "a", encoding="utf-8") as ferr:
                            ferr.write(json.dumps({
                                "ts_sec": time.time() - start_ts,
                                "id": out.get("id", row_id),
                                "error": out.get("_error", ""),
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                    with open(cpu_timing_csv, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, "0.000000", "0.000000", 0, 0, "error"])
                    if _env_flag("ONTOLOGY_VERIFY_STRICT", "1"):
                        abort_error = f"{out.get('id', row_id)}: {out.get('_error', '')}"
                        # Best-effort: stop remaining workers so we can exit the loop cleanly.
                        try:
                            cpu_queue.clear()
                        except Exception:
                            cpu_queue = []
                        for s2 in cpu_slots:
                            try:
                                c2 = s2.get("conn")
                                if c2:
                                    try:
                                        c2.close()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            try:
                                p2 = s2.get("proc")
                                if p2 and p2.is_alive():
                                    try:
                                        p2.terminate()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        cpu_slots = []
                        try:
                            for w2 in gpu_workers:
                                try:
                                    w2["busy"] = False
                                    w2["row_id"] = ""
                                    w2["row_payload"] = None
                                except Exception:
                                    pass
                                try:
                                    w2["inq"].put(None)
                                except Exception:
                                    pass
                            for w2 in gpu_workers:
                                try:
                                    w2["proc"].join(timeout=0.5)
                                except Exception:
                                    pass
                                if w2["proc"].is_alive():
                                    try:
                                        w2["proc"].terminate()
                                    except Exception:
                                        pass
                            gpu_workers_stopped = True
                        except Exception:
                            pass
                        done_gpu = total_gpu
                        done_cpu = total_cpu
                        break
                elif out:
                    with open(cpu_timing_csv, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([
                            out.get("id",""),
                            f'{out.get("t_filter",0.0):.6f}',
                            f'{out.get("t_match",0.0):.6f}',
                            out.get("cand_asts",0),
                            out.get("filtered_asts",0),
                            ""
                        ])
                    shards = out.get("shards") or {}
                    if shards:
                        _append_csv_from_jsonl(candidate_csv, triple_cols, shards.get("candidate_jsonl", ""))
                        if os.path.exists(candidate_slim_csv):
                            _append_csv_from_jsonl(candidate_slim_csv, slim_cols, shards.get("candidate_jsonl", ""))
                        _append_csv_from_jsonl(vis_csv, vis_cols, shards.get("vis_jsonl", ""))
                        _append_csv_from_jsonl(verified_csv, triple_cols, shards.get("verified_jsonl", ""))

                        # JSONL export of verified triples (subject=domain_arg, object=range_arg).
                        triples = []
                        seen_tr = set()
                        vpath = shards.get("verified_jsonl", "")
                        if vpath and os.path.exists(vpath):
                            with open(vpath, "r", encoding="utf-8") as fv:
                                for line in fv:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        r = json.loads(line)
                                    except Exception:
                                        continue
                                    sub = strip_trailing_particles((r.get("domain_arg") or ""))
                                    rel = (r.get("relation_ja") or "").strip()
                                    obj = strip_trailing_particles((r.get("range_arg") or ""))
                                    if not sub or not rel or not obj:
                                        continue
                                    key = (sub, rel, obj)
                                    if key in seen_tr:
                                        continue
                                    seen_tr.add(key)
                                    triples.append({"sub": sub, "rel": rel, "obj": obj})
                        try:
                            with open(extracted_jsonl_path, "a", encoding="utf-8") as fex:
                                fex.write(json.dumps({
                                    "id": out.get("id", ""),
                                    "sent_ja": out.get("sentence", ""),
                                    "extracted_triples": triples,
                                }, ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                        ppath = shards.get("prompt_jsonl", "")
                        if ppath and os.path.exists(ppath):
                            with open(prompt_log_path, "a", encoding="utf-8") as fpl, open(ppath, "r", encoding="utf-8") as fsrc:
                                for line in fsrc:
                                    if line.strip():
                                        fpl.write(line if line.endswith("\n") else (line + "\n"))

                        # Cleanup shards (best-effort)
                        for k in ("candidate_jsonl", "vis_jsonl", "verified_jsonl", "prompt_jsonl"):
                            pth = shards.get(k, "")
                            if pth and os.path.exists(pth):
                                try:
                                    os.remove(pth)
                                except Exception:
                                    pass
                    else:
                        # Backward-compat path (older workers sending full payloads).
                        candidate_rows = out.get("candidates", [])
                        if candidate_rows:
                            _append_csv_rows(candidate_csv, triple_cols, candidate_rows)
                            if os.path.exists(candidate_slim_csv):
                                _append_csv_rows(candidate_slim_csv, slim_cols, candidate_rows)
                        vis_rows = out.get("vis", [])
                        if vis_rows:
                            _append_csv_rows(vis_csv, vis_cols, vis_rows)
                        verified_rows = out.get("verified", [])
                        if verified_rows:
                            _append_csv_rows(verified_csv, triple_cols, verified_rows)
                        triples = []
                        seen_tr = set()
                        for r in (verified_rows or []):
                            sub = strip_trailing_particles((r.get("domain_arg") or ""))
                            rel = (r.get("relation_ja") or "").strip()
                            obj = strip_trailing_particles((r.get("range_arg") or ""))
                            if not sub or not rel or not obj:
                                continue
                            key = (sub, rel, obj)
                            if key in seen_tr:
                                continue
                            seen_tr.add(key)
                            triples.append({"sub": sub, "rel": rel, "obj": obj})
                        try:
                            with open(extracted_jsonl_path, "a", encoding="utf-8") as fex:
                                fex.write(json.dumps({
                                    "id": out.get("id", ""),
                                    "sent_ja": out.get("sentence", ""),
                                    "extracted_triples": triples,
                                }, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        prompt_rows = out.get("prompt_logs", [])
                        if prompt_rows:
                            with open(prompt_log_path, "a", encoding="utf-8") as fpl:
                                for row in prompt_rows:
                                    fpl.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    with open(cpu_timing_csv, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, "0.000000", "0.000000", 0, 0, "empty"])

                done_cpu += 1
                cpu_pbar.update(1)
            else:
                if time.time() - started > CPU_TOTAL_TIMEOUT_SEC:
                    try: p.terminate()
                    except Exception: pass
                    try: p.join(timeout=0.5)
                    except Exception: pass
                    with open(cpu_timing_csv, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, f'{CPU_TOTAL_TIMEOUT_SEC:.6f}', "0.000000", 0, 0, "timeout"])
                    done_cpu += 1
                    cpu_pbar.update(1)
                else:
                    still_cpu.append(slot)
        cpu_slots = [] if abort_error else still_cpu

        log_inflight()
        time.sleep(0.01)

    # Stop GPU workers (if not already stopped earlier)
    if not locals().get("gpu_workers_stopped", False):
        for w in gpu_workers:
            try:
                w["inq"].put(None)
            except Exception:
                pass
        for w in gpu_workers:
            try:
                w["proc"].join(timeout=1.0)
            except Exception:
                pass
            if w["proc"].is_alive():
                try:
                    w["proc"].terminate()
                except Exception:
                    pass

    gpu_pbar.close()
    cpu_pbar.close()

    if abort_error:
        raise RuntimeError(f"fail-fast (ONTOLOGY_VERIFY_STRICT=1): {abort_error}")

    elapsed = time.time() - start_ts
    print("抽出処理時間: {:.1f} 秒".format(elapsed))
    print("=== 抽出完了（逐次書き込み） ===")
    print("保存先(candidate): {}".format(candidate_csv))
    print("保存先(verified): {}".format(verified_csv))
    print("保存先(可視化): {}".format(vis_csv))
    print("ログ: {}".format(log_dir))
    summary_path, warn_lines, _summary = write_prompt_accept_summary(prompt_log_path, log_dir)
    if summary_path:
        print(f"prompt accept summary: {summary_path}")
    for w in (warn_lines or []):
        print(w)
def main():
    preflight_ontology_llm()
    print("パターンJSONをロード中…")
    patterns = load_and_compile_patterns(
        index_path=PATTERN_INDEX_JSON,
        jsonl_path=PATTERN_JSONL,
    )
    ast_dict = build_ast_dict(patterns)
    # Pre-build CKYMatcher objects once in the parent process.
    # CPU workers are forked per sentence, so this avoids rebuilding matchers hundreds of times.
    try:
        total_matchers = 0
        for _v, entries in ast_dict.items():
            for e in entries:
                if "matcher" not in e:
                    e["matcher"] = CKYMatcher(e["ast"], verbose=False)
                if "parallel_var_groups" not in e:
                    e["parallel_var_groups"] = extract_parallel_variable_groups(e["ast"])
                total_matchers += 1
        print(f"CKYMatcher 事前構築: {total_matchers} パターン")
    except Exception as e:
        print(f"[WARN] CKYMatcher 事前構築に失敗: {e}")
    print("ロード完了: {} パターン".format(len(patterns)))

    jsonl_paths = sorted(glob(os.path.join(INPUT_JSONL_DIR, "*.jsonl")))
    if not jsonl_paths:
        print(f"入力JSONLが見つかりません: {INPUT_JSONL_DIR}")
        return

    for path in jsonl_paths:
        process_jsonl(path, ast_dict)


# =============================================================
# エントリポイント
# =============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
