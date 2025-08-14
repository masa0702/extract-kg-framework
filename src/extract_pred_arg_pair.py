# extract_po_pair_main_fast_spawn.py
# =============================================================
# ① AST Pickle をロード
# ② GiNZA 依存解析 + CKY 表キャッシュ
# ③ 文ごとに **ProcessPool（spawn）** で CKYMatcher を適用
# ④ GPU が使えれば **子プロセス側で** BERT を cuda に載せる
# ⑤ tqdm で進捗表示
#     ＋ 各文の処理が終わるたびに CSV へ追記保存
# =============================================================

import os
import json
import gzip
import pickle
import time
import pandas as pd
from collections import defaultdict
from itertools import product
from tqdm.auto import tqdm

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# =============================================================
# 定数のみモジュール直下に置く（重い初期化は置かない）
# =============================================================
AST_PICKLE      = "../data/patterns/patterns_ast.pkl.gz"
INPUT_SENT_CSV  = "../data/target_datas/movie_target_data.csv"

dir_name   = os.path.basename(os.path.dirname(INPUT_SENT_CSV))
filename   = os.path.basename(INPUT_SENT_CSV)
prefix     = filename[:-4]
output_dir = f"../results/extract_pred_arg_pair/{dir_name}/{prefix}/"

dep_json_path  = f"{output_dir}{prefix}_dependency_analysis.json"
cky_json_path  = f"{output_dir}{prefix}_dependency_analysis_with_cky.json"
RESULT_CSV     = f"{output_dir}{prefix}_extract_po_pair.csv"

EXCLUDE_POS = ["助詞", "接続詞", "助動詞",
               "補助記号-句点", "補助記号-読点",
               "記号-句点", "記号-読点"]

# =============================================================
# グローバル（子プロセスでセットする）
# =============================================================
analyzer = None
cky_json_data_g = None
ast_dict_g = None
exclude_pos_g = None

# =============================================================
# ヘルパ
# =============================================================
def extract_parallel_variables(ast):
    """ParallelNode 直下の VariableNode のみ取得（内包表現なし）"""
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

    names = []
    for v in vars_:
        names.append(f"{v.symbol}{v.index}")
    return names

def clean_variable_mapping(varmap, clauses):
    """助詞等を落として再結合（内包表現なし）"""
    new_map = {}
    for var, val in varmap.items():
        found = None
        for cl in clauses:
            if cl[0] == val:
                found = cl
                break
            else:
                if val:
                    if isinstance(val, str):
                        if cl[0] in val:
                            found = cl
                            break
        if found:
            tokens = found[2]
            xpos   = found[4]
            filtered = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                pos = xpos[i]
                skip = False
                j = 0
                while j < len(EXCLUDE_POS):
                    if EXCLUDE_POS[j] in pos:
                        skip = True
                        break
                    j += 1
                if not skip:
                    filtered.append(tok)
                i += 1
            if len(filtered) > 0:
                new_map[var] = "".join(filtered)
        else:
            new_map[var] = val
    return new_map

# =============================================================
# 子プロセス初期化
# =============================================================
def init_worker(device_id, cky_json_data, ast_dict, exclude_pos):
    global analyzer
    global cky_json_data_g
    global ast_dict_g
    global exclude_pos_g

    cky_json_data_g = cky_json_data
    ast_dict_g = ast_dict
    exclude_pos_g = exclude_pos

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    # ★ここで BERT / CUDA を初期化
    try:
        local_analyzer = CKYAnalyzer()
    except TypeError:
        local_analyzer = CKYAnalyzer()

    if device.startswith("cuda") and hasattr(local_analyzer, "model"):
        local_analyzer.model.to(device)

    analyzer = local_analyzer

# =============================================================
# 文1件処理
# =============================================================
def process_sentence(row_dict):
    global analyzer
    global cky_json_data_g
    global ast_dict_g

    sent_id  = row_dict["id"]
    sentence = row_dict["sent"]

    if sentence not in cky_json_data_g:
        return []

    info = cky_json_data_g[sentence]
    cky_table = info["dependency_table"]
    clauses   = info["clauses"]

    cky_dep = analyzer.analyze_cky_table(cky_table)
    bunsetsu_cnt = len(clauses)

    candidate_asts = []
    v = 2  # exclude var_count=1
    while v <= bunsetsu_cnt:
        if v in ast_dict_g:
            lst = ast_dict_g[v]
            i = 0
            while i < len(lst):
                candidate_asts.append(lst[i])
                i += 1
        v += 1

    if len(candidate_asts) == 0:
        return []

    # --- Cell-based filtering ---
    filtered = []
    for entry in candidate_asts:
        var_count = entry.get("var_count", 0)
        literals = entry.get("literal_list", [])
        par_cnt = entry.get("parallel_var_count", 0)
        passed = False
        i1 = 0
        while i1 < bunsetsu_cnt and not passed:
            j1 = i1 + 1
            while j1 < bunsetsu_cnt:
                chunk_num = j1 - i1 + 1
                if var_count > chunk_num:
                    j1 += 1
                    continue
                if literals:
                    cell_text = CkyTable.get_cell_span_text(clauses, i1 + 1, j1 + 1)
                    lit_ok = True
                    k = 0
                    while k < len(literals):
                        if literals[k] not in cell_text:
                            lit_ok = False
                            break
                        k += 1
                    if not lit_ok:
                        j1 += 1
                        continue
                if par_cnt >= 2:
                    if CkyTable.count_parallel_keys(clauses, i1 + 1, j1 + 1) < par_cnt - 1:
                        j1 += 1
                        continue
                passed = True
                break
            i1 += 1
        if passed:
            filtered.append(entry)

    candidate_asts = filtered

    if len(candidate_asts) == 0:
        return []

    seen = set()
    recs = []

    i = 0
    while i < len(candidate_asts):
        entry = candidate_asts[i]
        ast = entry["ast"]
        matcher = CKYMatcher(ast, verbose=False)

        for r in matcher.match_table(cky_dep):
            key = frozenset(r.variable_mapping.items())
            if key in seen:
                continue
            seen.add(key)

            cmap = clean_variable_mapping(r.variable_mapping, clauses)

            par_names = extract_parallel_variables(ast)
            par_elems = []
            j = 0
            while j < len(par_names):
                name = par_names[j]
                if name in cmap:
                    par_elems.append(cmap[name])
                j += 1

            if len(par_elems) > 0:
                judge = judge_parallel(sentence, par_elems)
                if judge is False:
                    continue

            Xs = []
            Ys = []
            for k, v2 in cmap.items():
                if k.startswith("X"):
                    Xs.append((k, v2))
                elif k.startswith("Y"):
                    Ys.append((k, v2))
            Xs = list({xv: (xk, xv) for xk, xv in Xs}.values())
            Ys = list({yv: (yk, yv) for yk, yv in Ys}.values())

            if len(Xs) == 0 or len(Ys) == 0:
                continue

            for idx, ((xk, xv), (yk, yv)) in enumerate(product(Xs, Ys)):
                rec = {
                    "id":           sent_id,
                    "sentence":     sentence,
                    "triple_index": idx,
                    "rel_ja":       yv,
                    "arg_ja":       xv
                }
                recs.append(rec)
        i += 1

    return recs

# =============================================================
# main
# =============================================================
def main():
    print("AST Pickle をロード中…")
    with gzip.open(AST_PICKLE, "rb") as fp:
        patterns_ast = pickle.load(fp)

    ast_dict = defaultdict(list)
    i = 0
    while i < len(patterns_ast):
        entry = patterns_ast[i]
        ast = entry["ast"]
        entry["literal_list"] = extract_literal_strings(ast)
        entry["parallel_var_count"] = count_parallel_variables(ast)
        ast_dict[entry["var_count"]].append(entry)
        i += 1
    print("ロード完了: {} パターン".format(len(patterns_ast)))

    print("出力ディレクトリ作成: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("依存解析 + CKY 準備開始")

    sent_df = pd.read_csv(INPUT_SENT_CSV, dtype=str)

    # sentences = sent_df["sent"].unique().tolist() を使わずループで
    sentences = []
    for s in sent_df["sent"].unique():
        sentences.append(s)

    try:
        with open(dep_json_path, "r", encoding="utf-8") as f:
            dep_data = json.load(f)
        if not isinstance(dep_data, dict):
            dep_data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        dep_data = {}

    new_sentences = []
    j = 0
    while j < len(sentences):
        s = sentences[j]
        if s not in dep_data:
            new_sentences.append(s)
        j += 1

    depana  = DependencyAnalysis()
    myutil  = MyUtility()
    cky_obj = CkyTable()

    if len(new_sentences) > 0:
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

    # --- 空CSV作成 ---
    header_cols = ["id", "sentence", "triple_index", "rel_ja", "arg_ja"]
    pd.DataFrame(columns=header_cols).to_csv(
        RESULT_CSV, index=False, encoding="utf-8-sig"
    )

    start = time.time()

    max_workers = 20
    print("プロセス並列実行: workers={}".format(max_workers))

    # --- spawn 強制 ---
    ctx = mp.get_context("spawn")

    # --- 送信用 row_dict リスト化（Series は重いので辞書化）---
    rows = []
    for _, r in sent_df.iterrows():
        rows.append({"id": r["id"], "sent": r["sent"]})

    futures = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(0, cky_json_data, ast_dict, EXCLUDE_POS)
    ) as ex:
        i = 0
        while i < len(rows):
            futures.append(ex.submit(process_sentence, rows[i]))
            i += 1

        for f in tqdm(as_completed(futures), total=len(futures), desc="Sentences"):
            try:
                recs = f.result()
            except Exception as e:
                print("worker error:", e)
                continue

            if not recs:
                continue

            pd.DataFrame(recs).to_csv(
                RESULT_CSV,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig"
            )

    elapsed = time.time() - start
    print("抽出処理時間: {:.1f} 秒".format(elapsed))
    print("=== 抽出完了（逐次書き込み） ===")
    print("保存先: {}".format(RESULT_CSV))

# =============================================================
# エントリポイント
# =============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
