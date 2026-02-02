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
)

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
    os.path.join(REPO_ROOT, "data/T2KGB_JA/target_data"),
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

GPU_WORKERS            = 2
GPU_TIMEOUT_SEC        = 4000000000
CPU_WORKERS            = max(4, min(64, (os.cpu_count() or 8) - 2))
CPU_TOTAL_TIMEOUT_SEC  = 1000000000

LIT_MAX_FREQ            = 20
CAND_SPAN_LIMIT_PER_AST = 1000

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
    out = []
    for row in rows:
        if ont and normalize_text(row.get("ontology_id")) != ont:
            continue
        pred = normalize_text(row.get("predicate_ja"))
        if pred and pred in sentence:
            out.append(pred)
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
        found = None
        for cl in clauses:
            if cl[0] == val:
                found = cl
                break
            else:
                if val and isinstance(val, str) and cl[0] in val:
                    found = cl
                    break
        if found:
            tokens = found[2]
            xpos   = found[4]
            filtered = []
            for tok, pos in zip(tokens, xpos):
                if any(x in pos for x in EXCLUDE_POS):
                    continue
                filtered.append(tok)
            if filtered:
                new_map[var] = "".join(filtered)
        else:
            new_map[var] = val
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
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
        else:
            device = "cpu"

        analyzer = CKYAnalyzer()
        if device.startswith("cuda") and hasattr(analyzer, "model"):
            analyzer.model.to(device)

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
        }
        conn.send(out)
    except Exception as e:
        try:
            conn.send({"_error": str(e), "id": row_payload.get("id", "")})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def cpu_child_worker(payload, ast_dict, conn: Connection):
    try:
        judge = ParallelJudgeLLMJP()
        ontology_resolver = get_ontology_resolver(
            RELATION_PROMPT_MAP_JSON, PROMPTS_JSON, ONTOLOGY_DIR
        )
        ontology_judge = get_ontology_judge()
        ontology_id = payload.get("ontology_id") or ""
        sent_id  = payload["id"]; sentence = payload["sentence"]
        cky_dep  = payload["cky_dep"]; clauses = payload["clauses"]
        B = len(clauses)

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

        t1 = time.time()
        seen = set()
        candidates = []
        vis_rows = []
        verified = []
        seen_triples = set()
        prompt_logs = []

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

        for entry in filtered:
            ast = entry["ast"]
            matcher = CKYMatcher(ast, verbose=False)
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
                    candidates.append({
                        "id": sent_id,
                        "sentence": sentence,
                        "ontology_id": ontology_id,
                        "relation_ja": yv,
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
                    row = ontology_resolver.resolve_relation_row(rel, ontology_id)
                    if row:
                        resolved_rows[rel] = row

                if not resolved_rows:
                    fallback_rels = extract_relation_candidates_from_sentence(
                        sentence,
                        ontology_id,
                        ontology_resolver,
                    )
                    for rel in fallback_rels:
                        row = ontology_resolver.resolve_relation_row(rel, ontology_id)
                        if row:
                            resolved_rows[rel] = row

                for rel, row in resolved_rows.items():
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
                            prompt_text = render_prompt(
                                prompt,
                                {
                                    "relation_ja": rel,
                                    "domain_concept_ja": domain_concept,
                                    "range_concept_ja": range_concept,
                                    "arg1": arg1,
                                    "arg2": arg2,
                                    "context_sentence": sentence,
                                },
                            )
                            verdict = ontology_judge.judge_prompt(prompt_text)
                            prompt_logs.append({
                                "id": sent_id,
                                "relation_ja": rel,
                                "ontology_id": ontology_id,
                                "prompt_id": prompt_id,
                                "prompt_name": prompt_name,
                                "mode": "pair",
                                "arg1": arg1,
                                "arg2": arg2,
                                "verdict": verdict,
                                "prompt_text": prompt_text,
                            })

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

                            key = (sent_id, rel, domain_arg, range_arg, prompt_id)
                            if key in seen_triples:
                                continue
                            seen_triples.add(key)
                            verified.append({
                                "id": sent_id,
                                "sentence": sentence,
                                "ontology_id": ontology_id,
                                "relation_ja": rel,
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

                        def _judge_side(side: str, concept: str, argument: str, other_argument: str) -> int:
                            prompt_text = render_prompt(
                                prompt,
                                {
                                    "relation_ja": rel,
                                    "side": side,
                                    "concept_ja": concept,
                                    "argument": argument,
                                    "other_argument": other_argument or "NULL",
                                    "context_sentence": sentence,
                                },
                            )
                            v = ontology_judge.judge_prompt(prompt_text)
                            prompt_logs.append({
                                "id": sent_id,
                                "relation_ja": rel,
                                "ontology_id": ontology_id,
                                "prompt_id": prompt_id,
                                "prompt_name": prompt_name,
                                "mode": side,
                                "argument": argument,
                                "other_argument": other_argument or "NULL",
                                "verdict": v,
                                "prompt_text": prompt_text,
                            })
                            return v

                        for a, b in _iter_allowed_pairs(x_values, parallel_value_groups):
                            # Try (domain=a, range=b)
                            vd = _judge_side("domain", domain_concept, a, b)
                            if vd == 1:
                                vr = _judge_side("range", range_concept, b, a)
                                if vr == 1:
                                    domain_arg, range_arg = a, b
                                    verdict = 1
                                    key = (sent_id, rel, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified.append({
                                            "id": sent_id,
                                            "sentence": sentence,
                                            "ontology_id": ontology_id,
                                            "relation_ja": rel,
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
                                    key = (sent_id, rel, domain_arg, range_arg, prompt_id)
                                    if key not in seen_triples:
                                        seen_triples.add(key)
                                        verified.append({
                                            "id": sent_id,
                                            "sentence": sentence,
                                            "ontology_id": ontology_id,
                                            "relation_ja": rel,
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

        out = {
            "id": sent_id,
            "candidates": candidates,
            "vis": vis_rows,
            "verified": verified,
            "prompt_logs": prompt_logs,
            "t_filter": t_filter,
            "t_match": t_match,
            "cand_asts": len(candidate_asts),
            "filtered_asts": len(filtered)
        }
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

    sent_stats_csv = os.path.join(log_dir, f"{prefix}_sentence_stats.csv")
    gpu_timing_csv = os.path.join(log_dir, f"{prefix}_gpu_timing.csv")
    gpu_done_csv = os.path.join(log_dir, f"{prefix}_gpu_done.csv")
    gpu_timeout_csv = os.path.join(log_dir, f"{prefix}_gpu_timeout.csv")
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

    gpu_slots = []
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

    while (done_gpu < total_gpu) or gpu_slots or cpu_queue or cpu_slots:
        while len(gpu_slots) < GPU_WORKERS and gpu_idx < total_gpu:
            row = rows[gpu_idx]
            parent_conn, child_conn = ctx_gpu.Pipe(duplex=False)
            dev_id = len(gpu_slots) % GPU_WORKERS
            p = ctx_gpu.Process(target=gpu_child_worker, args=(row, dev_id, child_conn), daemon=True)
            p.start()
            submitted_gpu += 1
            gpu_slots.append({
                "proc": p,
                "conn": parent_conn,
                "start": time.time(),
                "device_id": dev_id,
                "row_id": row["id"],
            })
            gpu_idx += 1

        still_gpu = []
        for slot in gpu_slots:
            p: mp.Process = slot["proc"]
            conn: Connection = slot["conn"]
            row_id = slot["row_id"]
            started = slot["start"]

            if not p.is_alive():
                payload = None
                try:
                    if conn.poll():
                        payload = conn.recv()
                except EOFError:
                    payload = None
                finally:
                    try: conn.close()
                    except Exception: pass
                    p.join(timeout=0.1)

                with open(gpu_done_csv, "a", newline="", encoding="utf-8") as fgd:
                    csv.writer(fgd).writerow([f'{time.time()-start_ts:.3f}', row_id])

                if payload and "_error" in payload:
                    done_gpu += 1
                    gpu_pbar.update(1)
                elif payload:
                    with open(gpu_timing_csv, "a", newline="", encoding="utf-8") as fgpu:
                        csv.writer(fgpu).writerow([payload["id"],
                                                   f'{payload.get("t_analyze", 0.0):.6f}',
                                                   payload.get("payload_size", -1)])
                    cpu_queue.append(payload)
                    done_gpu += 1
                    gpu_pbar.update(1)
                else:
                    done_gpu += 1
                    gpu_pbar.update(1)
            else:
                if time.time() - started > GPU_TIMEOUT_SEC:
                    try: p.terminate()
                    except Exception: pass
                    try: p.join(timeout=0.5)
                    except Exception: pass
                    with open(gpu_timeout_csv, "a", newline="", encoding="utf-8") as fgt:
                        csv.writer(fgt).writerow([f'{time.time()-start_ts:.3f}', row_id])
                    done_gpu += 1
                    gpu_pbar.update(1)
                else:
                    still_gpu.append(slot)
        gpu_slots = still_gpu

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
                    if conn.poll():
                        out = conn.recv()
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
                    candidate_rows = out.get("candidates", [])
                    if candidate_rows:
                        pd.DataFrame(candidate_rows).to_csv(
                            candidate_csv, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                    vis_rows = out.get("vis", [])
                    if vis_rows:
                        pd.DataFrame(vis_rows).to_csv(
                            vis_csv, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                    verified_rows = out.get("verified", [])
                    if verified_rows:
                        pd.DataFrame(verified_rows).to_csv(
                            verified_csv, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                    # JSONL export of verified triples (subject=domain_arg, object=range_arg).
                    # One line per processed sentence/id for downstream evaluation.
                    triples = []
                    seen_tr = set()
                    for r in (verified_rows or []):
                        sub = (r.get("domain_arg") or "").strip()
                        rel = (r.get("relation_ja") or "").strip()
                        obj = (r.get("range_arg") or "").strip()
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
        cpu_slots = still_cpu

        log_inflight()
        time.sleep(0.01)

    gpu_pbar.close()
    cpu_pbar.close()

    elapsed = time.time() - start_ts
    print("抽出処理時間: {:.1f} 秒".format(elapsed))
    print("=== 抽出完了（逐次書き込み） ===")
    print("保存先(candidate): {}".format(candidate_csv))
    print("保存先(verified): {}".format(verified_csv))
    print("保存先(可視化): {}".format(vis_csv))
    print("ログ: {}".format(log_dir))
def main():
    print("パターンJSONをロード中…")
    patterns = load_and_compile_patterns(
        index_path=PATTERN_INDEX_JSON,
        jsonl_path=PATTERN_JSONL,
    )
    ast_dict = build_ast_dict(patterns)
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
