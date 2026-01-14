# extract_pred_arg_pair.py
# =============================================================
# ① AST Pickle をロード（ASTメタ付与: literal_list, parallel_var_count）
# ② GiNZA 依存解析 + CKY 表キャッシュを用意
# ③ メインプロセスが手動でパイプラインを監督
#     - GPUステージ（同時2本, 各文=子プロセス, device 0/1に割当）
#         → CKYAnalyzerで cky_dep を生成（30秒超でプロセスkillしてスキップ）
#     - CPUステージ（多本, 各文=子プロセス）
#         → フィルタ→候補生成→CKYMatcher（合算15秒でプロセスkill）
# ④ tqdm で進捗表示（結果は逐次CSV追記）
# ⑤ 診断ログ（sentence_stats / gpu_timing / gpu_done / gpu_timeout / cpu_timing / inflight）
# =============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import gzip
import pickle
import time
import csv
import tempfile
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm.auto import tqdm
from bisect import bisect_left, bisect_right

import multiprocessing as mp
from multiprocessing.connection import Connection
import signal

import torch

# ---------- 既存モジュール ----------
from pattern_nodes import (
    ParallelNode,
    VariableNode,
    extract_literal_strings,
    count_parallel_variables,
)
from matcher import CKYMatcher
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from clause_analysis import DependencyAnalysis
from utils import MyUtility
from semantic_judge import judge_parallel
from filter_settings import PARALLEL_KEYS

# =============================================================
# 定数
# =============================================================
AST_PICKLE      = "../data/patterns/swopatterns_ast.pkl.gz"
INPUT_SENT_CSV  = "../data/target_datas/swo_target_data.csv"

dir_name   = os.path.basename(os.path.dirname(INPUT_SENT_CSV))
filename   = os.path.basename(INPUT_SENT_CSV)
prefix     = filename[:-4]
output_dir = f"../results/extract_pred_arg_pair/{dir_name}/{prefix}/"

dep_json_path  = f"{output_dir}{prefix}_dependency_analysis.json"
cky_json_path  = f"{output_dir}{prefix}_dependency_analysis_with_cky.json"
RESULT_CSV     = f"{output_dir}{prefix}_extract_po_pair.csv"

# 診断ログの保存先
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

# ---- 制限・スロット数 ----
GPU_WORKERS            = 2             # 同時GPUスロット（CUDA_VISIBLE_DEVICES=0,1 を想定）
GPU_TIMEOUT_SEC        = 4000000000            # 1文のGPU解析ウォッチドッグ
CPU_WORKERS            = max(4, min(64, (os.cpu_count() or 8) - 2))  # 同時CPUスロット
CPU_TOTAL_TIMEOUT_SEC  = 1000000000            # 1文のCPU（フィルタ+マッチング）合算上限

# ---- フィルタ高速化パラメタ ----
LIT_MAX_FREQ            = 20           # 高頻度リテラルはフィルタ判定から除外
CAND_SPAN_LIMIT_PER_AST = 1000         # ASTごとの候補(i,j)上限

# =============================================================
# ユーティリティ（共通）
# =============================================================
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
            i = 0
            while i < len(tokens):
                tok = tokens[i]; pos = xpos[i]
                skip = False
                j = 0
                while j < len(EXCLUDE_POS):
                    if EXCLUDE_POS[j] in pos:
                        skip = True; break
                    j += 1
                if not skip:
                    filtered.append(tok)
                i += 1
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

# =============================================================
# GPU 子プロセス：1文のCKY解析
#   入力: row_payload = {"id","sent","cky_table","clauses"}
#   出力: Pipeで dict を送信 or 何も送らない（タイムアウト時は親がkill）
# =============================================================
def gpu_child_worker(row_payload, device_id: int, conn: Connection):
    try:
        # CUDAデバイス設定
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
        else:
            device = "cpu"

        # アナライザ初期化（子プロセス毎）
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

# =============================================================
# CPU 子プロセス：1文のフィルタ+マッチング（合算15秒以内）
#   入力: payload = {"id","sentence","cky_dep","clauses"}
#   出力: Pipeで dict を送信（タイムアウトは親側がkillしfallback記録）
# =============================================================
def cpu_child_worker(payload, ast_dict, conn: Connection):
    try:
        sent_id  = payload["id"]; sentence = payload["sentence"]
        cky_dep  = payload["cky_dep"]; clauses = payload["clauses"]
        B = len(clauses)

        # 候補AST（var_count>=2）
        candidate_asts = []
        v = 2
        while v <= B:
            if v in ast_dict:
                candidate_asts.extend(ast_dict[v])
            v += 1

        # 文テキスト＆境界
        sent_text, starts = build_sentence_text_and_offsets(clauses)
        total_len = len(sent_text)
        full_start, full_end = 0, total_len

        # 事前インデクス
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

        # 粗フィルタ（全文）
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
        cand_asts_count = len(candidate_asts)

        # リテラル主導で候補セル生成（セル総当たり廃止）
        t0 = time.time()
        filtered = []

        for entry in candidate_asts:
            var_count = entry.get("var_count", 0)
            literals  = entry.get("literal_list", [])
            par_cnt   = entry.get("parallel_var_count", 0)

            # 実効リテラル（高頻度除外）
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

            # 候補 (i,j)
            cand_ij = []
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
                    cand_ij.append((i, j))
                    if i > 0: cand_ij.append((i - 1, j))
                    if j < B - 1: cand_ij.append((i, j + 1))
            else:
                cand_ij = [(0, B - 1)]

            # 判定
            passed = False
            seen_ij = set()
            for (i, j) in cand_ij:
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
                filtered.append(entry)

        t_filter = time.time() - t0

        # マッチング（時間制限は親が管理しkillするため、ここは素直に実行）
        t1 = time.time()
        seen = set(); recs = []

        for entry in filtered:
            ast = entry["ast"]
            matcher = CKYMatcher(ast, verbose=False)
            for r in matcher.match_table(cky_dep):
                key = frozenset(r.variable_mapping.items())
                if key in seen: continue
                seen.add(key)

                cmap = clean_variable_mapping(r.variable_mapping, clauses)

                par_names = extract_parallel_variables(ast)
                par_elems = [cmap[name] for name in par_names if name in cmap]
                if par_elems:
                    judge = judge_parallel(sentence, par_elems)
                    if judge is False:
                        continue

                Xs = []; Ys = []
                for k, v2 in cmap.items():
                    if k.startswith("X"): Xs.append((k, v2))
                    elif k.startswith("Y"): Ys.append((k, v2))
                Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
                Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())
                if not Xs or not Ys: continue

                for idx, ((xk, xv), (yk, yv)) in enumerate(product(Xs, Ys)):
                    recs.append({
                        "id":           sent_id,
                        "sentence":     sentence,
                        "triple_index": idx,
                        "rel_ja":       yv,
                        "arg_ja":       xv
                    })

        t_match = time.time() - t1

        out = {
            "id": sent_id,
            "recs": recs,
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

    # ASTメタ付与
    ast_dict = defaultdict(list)
    for entry in patterns_ast:
        ast = entry["ast"]
        entry["literal_list"] = extract_literal_strings(ast)
        entry["parallel_var_count"] = count_parallel_variables(ast)
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
    total_cpu = total_gpu  # GPU完了→CPU投入（GPU timeout分はCPU対象外だがバーは同じ総数で進める）

    # 結果CSVのヘッダ
    header_cols = ["id", "sentence", "triple_index", "rel_ja", "arg_ja"]
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=header_cols).to_csv(
            RESULT_CSV, index=False, encoding="utf-8-sig"
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

    # 文ごとの重さ指標（cells/並列総数の見積もり）
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

    # ast_dict を fork 共有させるため、親のグローバルにも置く
    global_ast_dict = ast_dict  # 参照名だけ

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

    # 先に軽い文から（cells小さい順）
    rows.sort(key=lambda r: (len(r["clauses"]) * (len(r["clauses"])-1)) // 2)

    gpu_pbar = tqdm(total=total_gpu, desc="GPU stage")
    cpu_pbar = tqdm(total=total_cpu, desc="CPU stage")

    # CPU投入待ちキュー（GPU完了payload）
    cpu_queue = []

    # ヘルパ：inflightログ
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
        # ---- GPU 起動補充（2スロット）----
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
                # 終了
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

                # ログ
                with open(GPU_DONE_CSV, "a", newline="", encoding="utf-8") as fgd:
                    csv.writer(fgd).writerow([f'{time.time()-start_ts:.3f}', row_id])

                # エラーハンドリング
                if payload and "_error" in payload:
                    # エラー扱いとしてスキップ
                    done_gpu += 1
                    gpu_pbar.update(1)
                elif payload:
                    # GPU計測
                    with open(GPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fgpu:
                        csv.writer(fgpu).writerow([payload["id"],
                                                   f'{payload.get("t_analyze", 0.0):.6f}',
                                                   payload.get("payload_size", -1)])
                    cpu_queue.append(payload)
                    done_gpu += 1
                    gpu_pbar.update(1)
                else:
                    # 何も返らず終了→スキップ
                    done_gpu += 1
                    gpu_pbar.update(1)
            else:
                # まだ動作中→タイムアウト判定
                if time.time() - started > GPU_TIMEOUT_SEC:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    try:
                        p.join(timeout=0.5)
                    except Exception:
                        pass
                    with open(GPU_TIMEOUT_CSV, "a", newline="", encoding="utf-8") as fgt:
                        csv.writer(fgt).writerow([f'{time.time()-start_ts:.3f}', row_id])
                    done_gpu += 1
                    gpu_pbar.update(1)
                else:
                    still_gpu.append(slot)
        gpu_slots = still_gpu

        # ---- CPU 起動補充（Nスロット）----
        while len(cpu_slots) < CPU_WORKERS and cpu_queue:
            payload = cpu_queue.pop(0)
            parent_conn, child_conn = ctx_cpu.Pipe(duplex=False)
            # spawn だと ast_dict を引数で渡す必要、fork なら共有。両対応のため渡す。
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
            sentence = slot["sentence"]
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

                # ログ/結果
                if out and "_error" in out:
                    # エラー行として記録
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
                        # 逐次追記
                        pd.DataFrame(recs).to_csv(
                            RESULT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                else:
                    # 子が何も返さず終了（稀）→空記録
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, "0.000000", "0.000000", 0, 0, "empty"])

                done_cpu += 1
                cpu_pbar.update(1)
            else:
                # タイムアウト監視（合算15秒）
                if time.time() - started > CPU_TOTAL_TIMEOUT_SEC:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    try:
                        p.join(timeout=0.5)
                    except Exception:
                        pass
                    # タイムアウト行
                    with open(CPU_TIMING_CSV, "a", newline="", encoding="utf-8") as fcpu:
                        csv.writer(fcpu).writerow([row_id, f'{CPU_TOTAL_TIMEOUT_SEC:.6f}', "0.000000", 0, 0, "timeout"])
                    done_cpu += 1
                    cpu_pbar.update(1)
                else:
                    still_cpu.append(slot)
        cpu_slots = still_cpu

        # inflight定期ログ
        log_inflight()

        # 小休止（忙待ち抑制）
        time.sleep(0.01)

    gpu_pbar.close()
    cpu_pbar.close()

    elapsed = time.time() - start_ts
    print("抽出処理時間: {:.1f} 秒".format(elapsed))
    print("=== 抽出完了（逐次書き込み） ===")
    print("保存先: {}".format(RESULT_CSV))
    print("ログ: {}".format(LOG_DIR))


# =============================================================
# エントリポイント
# =============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # GPU子の安全性を優先
    main()
