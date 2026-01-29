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
import gzip
import pickle
import time
import csv
import hashlib
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm.auto import tqdm
from bisect import bisect_left, bisect_right

import multiprocessing as mp
from multiprocessing.connection import Connection

import torch

from pattern.pattern_nodes import (
    ParallelNode,
    VariableNode,
    extract_literal_strings,
    count_parallel_variables,
)
from modules_core.matcher import CKYMatcher
from modules_core.cky_table import CkyTable
from modules_bert.bert_modules import CKYAnalyzer
from modules_core.bunsetu import DependencyAnalysis
from modules_core.utils import MyUtility
from llm.parallel_judge import ParallelJudgeLLMJP
from config.filter_settings import PARALLEL_KEYS

AST_PICKLE      = "../data/patterns/swopatterns_ast.pkl.gz"
INPUT_SENT_CSV  = "../data/target_datas/swo_target_data.csv"

dir_name   = os.path.basename(os.path.dirname(INPUT_SENT_CSV))
filename   = os.path.basename(INPUT_SENT_CSV)
prefix     = filename[:-4]
output_dir = f"../results/extract_pred_arg_pair/{dir_name}/{prefix}/"

dep_json_path  = f"{output_dir}{prefix}_dependency_analysis.json"
cky_json_path  = f"{output_dir}{prefix}_dependency_analysis_with_cky.json"
RESULT_CSV     = f"{output_dir}{prefix}_extract_po_pair.csv"
VIS_CSV        = f"{output_dir}{prefix}_ast_visualization.csv"

LOG_DIR = os.path.join(output_dir, "logs")
SENT_STATS_CSV = os.path.join(LOG_DIR, f"{prefix}_sentence_stats.csv")
GPU_TIMING_CSV = os.path.join(LOG_DIR, f"{prefix}_gpu_timing.csv")
GPU_DONE_CSV   = os.path.join(LOG_DIR, f"{prefix}_gpu_done.csv")
GPU_TIMEOUT_CSV= os.path.join(LOG_DIR, f"{prefix}_gpu_timeout.csv")
CPU_TIMING_CSV = os.path.join(LOG_DIR, f"{prefix}_cpu_timing.csv")
INFLIGHT_CSV   = os.path.join(LOG_DIR, f"{prefix}_inflight.csv")

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
        recs = []
        vis_rows = []

        for entry in filtered:
            ast = entry["ast"]
            matcher = CKYMatcher(ast, verbose=False)

            for r in matcher.match_table(cky_dep, spans=entry.get("cand_spans")):
                key = frozenset(r.variable_mapping.items())
                if key in seen:
                    continue
                seen.add(key)

                varmap_raw   = dict(r.variable_mapping)
                varmap_clean = clean_variable_mapping(varmap_raw, clauses)

                par_names = extract_parallel_variables(ast)
                par_elems = [varmap_clean[name] for name in par_names if name in varmap_clean]
                if par_elems:
                    is_parallel = judge.judge_parallel(sentence, par_elems)
                    if is_parallel is False:
                        continue

                Xs = []; Ys = []
                for k, v2 in varmap_clean.items():
                    if k.startswith("X"): Xs.append((k, v2))
                    elif k.startswith("Y"): Ys.append((k, v2))
                Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
                Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())
                if not Xs or not Ys:
                    continue

                for idx, ((xk, xv), (yk, yv)) in enumerate(product(Xs, Ys)):
                    recs.append({
                        "id":           sent_id,
                        "sentence":     sentence,
                        "triple_index": idx,
                        "rel_ja":       yv,
                        "arg_ja":       xv
                    })

                ast_uid  = entry.get("ast_uid", get_ast_uid(ast))
                var_cnt  = entry.get("var_count", 0)
                par_cnt  = entry.get("parallel_var_count", 0)
                literals = entry.get("literal_list", [])

                vis_rows.append({
                    "id": sent_id,
                    "sentence": sentence,
                    "ast_uid": ast_uid,
                    "var_count": var_cnt,
                    "parallel_var_count": par_cnt,
                    "literals": "|".join(literals) if literals else "",
                    "X_vars": "|".join([f"{k}:{v}" for k,v in Xs]) if Xs else "",
                    "Y_vars": "|".join([f"{k}:{v}" for k,v in Ys]) if Ys else "",
                    "varmap_raw": json.dumps(varmap_raw, ensure_ascii=False),
                    "varmap_clean": json.dumps(varmap_clean, ensure_ascii=False),
                    "parallel_var_names": json.dumps(par_names, ensure_ascii=False),
                    "parallel_elements": json.dumps(par_elems, ensure_ascii=False),
                })

        t_match = time.time() - t1

        out = {
            "id": sent_id,
            "recs": recs,
            "vis": vis_rows,
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
def main():
    print("AST Pickle をロード中…")
    with gzip.open(AST_PICKLE, "rb") as fp:
        patterns_ast = pickle.load(fp)

    # ASTメタ付与（★ ast_uid 追加）
    ast_dict = defaultdict(list)
    for entry in patterns_ast:
        ast = entry["ast"]
        entry["literal_list"] = extract_literal_strings(ast)
        entry["parallel_var_count"] = count_parallel_variables(ast)
        entry["ast_uid"] = get_ast_uid(ast)  # ★追加
        ast_dict[entry["var_count"]].append(entry)
    print("ロード完了: {} パターン".format(len(patterns_ast)))

    # 出力ディレクトリ
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("依存解析 + CKY 準備開始")
    sent_df = pd.read_csv(INPUT_SENT_CSV, dtype=str)
    sentences = [s for s in sent_df["sent"].unique()]

    # 依存解析キャッシュ
    try:
        with open(dep_json_path, "r", encoding="utf-8") as f:
            dep_data = json.load(f)
        if not isinstance(dep_data, dict):
            dep_data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        dep_data = {}

    new_sentences = [s for s in sentences if s not in dep_data]

    depana  = DependencyAnalysis()
    myutil  = MyUtility()
    cky_obj = CkyTable()

    if new_sentences:
        print("GiNZA 解析: {} 文".format(len(new_sentences)))
        dep_results = depana.analyze_sentences(new_sentences)
        dep_data.update(dep_results)
        myutil.save_json_from_file(dep_data, dep_json_path)

    if not os.path.exists(cky_json_path):
        print("CKY 表を生成中 …")
        cky_obj.process_json_to_cky_and_save(dep_json_path, cky_json_path)

    with open(cky_json_path, "r", encoding="utf-8") as f:
        cky_json_data = json.load(f)

    print("依存解析 + CKY 準備完了")

    # 対象行（cky情報がある文のみ）
    rows = []
    for _, r in sent_df.iterrows():
        s = r["sent"]
        if s in cky_json_data:
            info = cky_json_data[s]
            rows.append({
                "id":   r["id"],
                "sent": s,
                "cky_table": info["dependency_table"],
                "clauses":   info["clauses"],
            })
    total_gpu = len(rows)
    total_cpu = total_gpu

    # 結果CSV（既存）
    header_cols = ["id", "sentence", "triple_index", "rel_ja", "arg_ja"]
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=header_cols).to_csv(
            RESULT_CSV, index=False, encoding="utf-8-sig"
        )

    # ★ 可視化CSV（新規）
    vis_cols = [
        "id","sentence","ast_uid","var_count","parallel_var_count","literals",
        "X_vars","Y_vars","varmap_raw","varmap_clean","parallel_var_names","parallel_elements"
    ]
    if not os.path.exists(VIS_CSV):
        pd.DataFrame(columns=vis_cols).to_csv(
            VIS_CSV, index=False, encoding="utf-8-sig"
        )

    # ログCSVのヘッダ
    if not os.path.exists(SENT_STATS_CSV):
        with open(SENT_STATS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","sentence_len","bunsetsu_cnt","cells","parallel_sum_all","cand_ast_estimate"])
    if not os.path.exists(GPU_TIMING_CSV):
        with open(GPU_TIMING_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","t_analyze_sec","payload_size_bytes"])
    if not os.path.exists(GPU_DONE_CSV):
        with open(GPU_DONE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","id"])
    if not os.path.exists(GPU_TIMEOUT_CSV):
        with open(GPU_TIMEOUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","id"])
    if not os.path.exists(CPU_TIMING_CSV):
        with open(CPU_TIMING_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","t_filter_sec","t_match_sec","cand_asts","filtered_asts","timeout"])
    if not os.path.exists(INFLIGHT_CSV):
        with open(INFLIGHT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_sec","inflight_gpu","inflight_cpu","done_gpu","done_cpu","submitted_gpu","submitted_cpu"])

    # 文ごとの重さ指標
    with open(SENT_STATS_CSV, "a", newline="", encoding="utf-8") as fstats:
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

    # ast_dict を親に保持（spawn/fork両対応）
    global_ast_dict = ast_dict

    # --------- 監督ループ開始 ---------
    start_ts = time.time()

    # GPU/CPU スロット
    ctx_gpu = mp.get_context("spawn")
    try:
        ctx_cpu = mp.get_context("fork")
    except ValueError:
        ctx_cpu = mp.get_context("spawn")

    gpu_slots = []   # list of dict(proc, parent_conn, start, device_id, row_id)
    cpu_slots = []   # list of dict(proc, parent_conn, start, row_id)

    gpu_idx = 0
    done_gpu = 0
    submitted_gpu = 0

    submitted_cpu = 0
    done_cpu = 0

    # 軽い文から処理
    rows.sort(key=lambda r: (len(r["clauses"]) * (len(r["clauses"])-1)) // 2)

    gpu_pbar = tqdm(total=total_gpu, desc="GPU stage")
    cpu_pbar = tqdm(total=total_cpu, desc="CPU stage")

    # CPU投入待ちキュー（GPU完了payload）
    cpu_queue = []

    # inflight ログ
    last_inflight_log = time.time()
    def log_inflight():
        nonlocal last_inflight_log
        now = time.time()
        if now - last_inflight_log >= 5.0:
            inflight_gpu = submitted_gpu - done_gpu
            inflight_cpu = submitted_cpu - done_cpu
            with open(INFLIGHT_CSV, "a", newline="", encoding="utf-8") as finf:
                csv.writer(finf).writerow([f'{now - start_ts:.3f}', inflight_gpu, inflight_cpu,
                                           done_gpu, done_cpu, submitted_gpu, submitted_cpu])
            last_inflight_log = now

    # メインイベントループ
    while (done_gpu < total_gpu) or gpu_slots or cpu_queue or cpu_slots:
        # ---- GPU 起動補充 ----
        while len(gpu_slots) < GPU_WORKERS and gpu_idx < total_gpu:
            row = rows[gpu_idx]
            parent_conn, child_conn = ctx_gpu.Pipe(duplex=False)
            dev_id = len(gpu_slots) % GPU_WORKERS  # 0/1を交互
            p = ctx_gpu.Process(target=gpu_child_worker, args=(row, dev_id, child_conn), daemon=True)
            p.start()
            submitted_gpu += 1
            gpu_slots.append({
                "proc": p,
                "conn": parent_conn,
                "start": time.time(),
                "device_id": dev_id,
                "row_id": row["id"]
            })
            gpu_idx += 1

        # ---- GPU 完了/タイムアウトチェック ----
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

                with open(GPU_DONE_CSV, "a", newline="", encoding="utf-8") as fgd:
                    csv.writer(fgd).writerow([f'{time.time()-start_ts:.3f}', row_id])

                if payload and "_error" in payload:
                    done_gpu += 1
                    gpu_pbar.update(1)
                elif payload:
                    with open(GPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fgpu:
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
                    with open(GPU_TIMEOUT_CSV, "a", newline="", encoding="utf-8") as fgt:
                        csv.writer(fgt).writerow([f'{time.time()-start_ts:.3f}', row_id])
                    done_gpu += 1
                    gpu_pbar.update(1)
                else:
                    still_gpu.append(slot)
        gpu_slots = still_gpu

        # ---- CPU 起動補充 ----
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
                "sentence": payload["sentence"]
            })

        # ---- CPU 完了/タイムアウトチェック ----
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

                # CPUログ/結果
                if out and "_error" in out:
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, "0.000000", "0.000000", 0, 0, "error"])
                elif out:
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([
                            out.get("id",""),
                            f'{out.get("t_filter",0.0):.6f}',
                            f'{out.get("t_match",0.0):.6f}',
                            out.get("cand_asts",0),
                            out.get("filtered_asts",0),
                            ""
                        ])
                    recs = out.get("recs", [])
                    if recs:
                        pd.DataFrame(recs).to_csv(
                            RESULT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                    # ★ 可視化行の追記
                    vis_rows = out.get("vis", [])
                    if vis_rows:
                        pd.DataFrame(vis_rows).to_csv(
                            VIS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                else:
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, "0.000000", "0.000000", 0, 0, "empty"])

                done_cpu += 1
                cpu_pbar.update(1)
            else:
                if time.time() - started > CPU_TOTAL_TIMEOUT_SEC:
                    try: p.terminate()
                    except Exception: pass
                    try: p.join(timeout=0.5)
                    except Exception: pass
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, f'{CPU_TOTAL_TIMEOUT_SEC:.6f}', "0.000000", 0, 0, "timeout"])
                    done_cpu += 1
                    cpu_pbar.update(1)
                else:
                    still_cpu.append(slot)
        cpu_slots = still_cpu

        # inflight定期ログ
        log_inflight()

        # 小休止
        time.sleep(0.01)

    gpu_pbar.close()
    cpu_pbar.close()

    elapsed = time.time() - start_ts
    print("抽出処理時間: {:.1f} 秒".format(elapsed))
    print("=== 抽出完了（逐次書き込み） ===")
    print("保存先(抽出): {}".format(RESULT_CSV))
    print("保存先(可視化): {}".format(VIS_CSV))
    print("ログ: {}".format(LOG_DIR))


# =============================================================
# エントリポイント
# =============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
