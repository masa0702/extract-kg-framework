# extract_ast_stats.py
# =============================================================
# ① AST Pickle をロード
# ② GiNZA 依存解析 + CKY 表キャッシュ（既存の前処理を流用）
# ③ 文ごとに「ASTフィルタ（var_count ≤ 文節数）」を適用した通過数を計測
# ④ 逐次CSVに追記保存 + 最後にJSONで全体集計サマリを保存
# ⑤ tqdmで進捗表示
# =============================================================

import os
import json
import gzip
import pickle
import time
import pandas as pd
from collections import defaultdict, Counter
from tqdm.auto import tqdm

# ---------- 既存モジュール（依存解析とCKY表生成のみ使用） ----------
from cky_table import CkyTable
from clause_analysis import DependencyAnalysis
from utils import MyUtility

# 追加インポート: ASTメタ情報とフィルタ設定
from pattern_nodes import extract_literal_strings, count_parallel_variables
from filter_settings import PARALLEL_KEYS

# =============================================================
# 定数
# =============================================================
AST_PICKLE      = "../data/patterns/patterns_ast.pkl.gz"
INPUT_SENT_CSV  = "../data/target_datas/culture_target_data.csv"

dir_name   = os.path.basename(os.path.dirname(INPUT_SENT_CSV))
filename   = os.path.basename(INPUT_SENT_CSV)
prefix     = filename[:-4]
output_dir = f"../results/extract_pred_arg_pair/{dir_name}/{prefix}/"

dep_json_path    = f"{output_dir}{prefix}_dependency_analysis.json"
cky_json_path    = f"{output_dir}{prefix}_dependency_analysis_with_cky.json"
RESULT_STATS_CSV = f"{output_dir}{prefix}_ast_stats_per_sentence.csv"
SUMMARY_JSON     = f"{output_dir}{prefix}_ast_stats_summary.json"

# =============================================================
# ASTユーティリティ
# =============================================================
try:
    from pattern_nodes import ParallelNode
except Exception:
    ParallelNode = None  # ParallelNode検出ができない場合は無効化

def has_parallel(node) -> bool:
    """ASTにParallelNodeを含むか（浅い/深い両方）。ParallelNode未インポートならFalse。"""
    if ParallelNode is None:
        return False

    found = False

    def visit(n):
        nonlocal found
        if found:
            return
        if isinstance(n, ParallelNode):
            found = True
            return
        for attr in ("elements", "options", "block"):
            if hasattr(n, attr):
                child = getattr(n, attr)
                if child is None:
                    continue
                if isinstance(child, list):
                    for c in child:
                        visit(c)
                else:
                    visit(child)

    visit(node)
    return found

# =============================================================
# メイン
# =============================================================
def main():
    print("AST Pickle をロード中…")
    with gzip.open(AST_PICKLE, "rb") as fp:
        patterns_ast = pickle.load(fp)
    print(f"ロード完了: {len(patterns_ast)} パターン")

    # ASTエントリのメタ情報を作成する
    # patterns_ast の各要素に literal_list と parallel_var_count を追加し、
    # また var_count ごとのバケットや ParallelNode 含有情報も保持する
    ast_dict = defaultdict(list)  # var_count -> list of ASTエントリ
    varcount_parallel_flags = defaultdict(list)  # var_count -> [bool,...] (ParallelNode含有) 
    # Stage2, Stage3 用に var_count ごとのAST総数カウンタを作成
    ast_entries = []
    for entry in patterns_ast:
        ast = entry["ast"]
        # ASTのリテラルと並列変数数を計算
        try:
            literal_list = extract_literal_strings(ast)
        except Exception:
            literal_list = []
        try:
            parallel_var_count = count_parallel_variables(ast)
        except Exception:
            parallel_var_count = 0
        # ASTエントリに追加
        v = entry["var_count"]
        entry["literal_list"] = literal_list
        entry["parallel_var_count"] = parallel_var_count
        ast_dict[v].append(entry)
        varcount_parallel_flags[v].append(has_parallel(ast))
        ast_entries.append(entry)

    print("出力ディレクトリ作成:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 入力CSV
    sent_df = pd.read_csv(INPUT_SENT_CSV, dtype=str)
    sentences = list(sent_df["sent"].unique())

    # 依存解析キャッシュを読む/更新
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
        print(f"GiNZA 解析: {len(new_sentences)} 文")
        dep_results = depana.analyze_sentences(new_sentences)
        dep_data.update(dep_results)
        myutil.save_json_from_file(dep_data, dep_json_path)

    if not os.path.exists(cky_json_path):
        print("CKY 表を生成中 …")
        cky_obj.process_json_to_cky_and_save(dep_json_path, cky_json_path)

    with open(cky_json_path, "r", encoding="utf-8") as f:
        cky_json_data = json.load(f)

    print("依存解析 + CKY 準備完了")

    # --- 空CSV作成（文ごとの統計を逐次追記） ---
    per_sentence_cols = [
        "id",
        "sentence",
        "bunsetsu_count",
        "ast_pool_total",            # var_count ≤ 文節数 を満たすAST総数
        "ast_pool_parallel",         # うちParallelNode含有AST数
        "ast_pool_nonparallel",      # うちParallelNode非含有AST数
        "by_varcount_json",          # var_count別AST数（JSON文字列）
        "by_varcount_parallel_json", # var_count別Parallel含有AST数（JSON文字列）
        "filter_stage1_pass",        # = ast_pool_total（初期段階では同値）
        "filter_stage2_pass",        # 予備（初期は同値）
        "filter_stage3_pass"         # 予備（初期は同値）
    ]
    pd.DataFrame(columns=per_sentence_cols).to_csv(
        RESULT_STATS_CSV, index=False, encoding="utf-8-sig"
    )

    # --- 全体集計用カウンタ ---
    total_docs = 0
    # var_count -> 総AST数（無条件）
    corpus_varcount_total = Counter()
    # Stage1: var_count ≤ 文節数 の通過AST数
    corpus_varcount_stage1 = Counter()
    # Stage2: リテラルフィルタ通過AST数
    corpus_varcount_stage2 = Counter()
    # Stage3: 並列数フィルタ通過AST数
    corpus_varcount_stage3 = Counter()
    # Stage1でParallelNodeを含むAST数
    corpus_varcount_parallel_stage1 = Counter()
    # Stage2/Stage3のParallelNode含有AST数（参考）
    corpus_varcount_parallel_stage2 = Counter()
    corpus_varcount_parallel_stage3 = Counter()
    bunsetsu_hist = Counter()               # 文節数分布
    stage1_hist = Counter()                # Stage1通過AST数（文単位）
    stage2_hist = Counter()                # Stage2通過AST数（文単位）
    stage3_hist = Counter()                # Stage3通過AST数（文単位）

    start = time.time()

    # Seriesの反復は遅いのでdictリスト化
    rows = [{"id": r["id"], "sent": r["sent"]} for _, r in sent_df.iterrows()]

    # コーパス総AST数（var_count無条件合算）
    for v, lst in ast_dict.items():
        corpus_varcount_total[v] += len(lst)

    print("文ごとのAST統計を計測中 …")
    for row in tqdm(rows, total=len(rows), desc="Sentences"):
        sent_id  = row["id"]
        sentence = row["sent"]

        # CKY情報がない文はスキップ
        info = cky_json_data.get(sentence)
        if not info:
            continue

        cky_table = info["dependency_table"]
        bunsetsu_cnt = len(cky_table[0]) if cky_table and len(cky_table[0]) > 0 else 0
        # 文字列処理に使用する文節リスト
        clauses = info.get("clauses", [])
        bunsetsu_hist[bunsetsu_cnt] += 1
        total_docs += 1

        # --- フィルタリング ---
        # Stage1: bunsetsu フィルタ (var_count ≤ bunsetsu_count)
        stage1_candidates = []
        by_v = {}
        by_v_parallel = {}
        stage1_total = 0
        stage1_parallel = 0
        for v in sorted(ast_dict.keys()):
            if v > bunsetsu_cnt:
                continue
            ast_list = ast_dict[v]
            # var_countが範囲内のASTはStage1候補
            by_v[v] = len(ast_list)
            # ParallelNode含有の数を計算
            parallel_flags = varcount_parallel_flags[v]
            count_v_parallel = sum(1 for f in parallel_flags if f)
            by_v_parallel[v] = count_v_parallel
            stage1_total += len(ast_list)
            stage1_parallel += count_v_parallel
            # append ASTエントリ
            stage1_candidates.extend(ast_list)
            # 累積カウンタ更新
            corpus_varcount_stage1[v] += len(ast_list)
            corpus_varcount_parallel_stage1[v] += count_v_parallel

        # Stage2: literal フィルタ
        # セル全体の文字列を取得（i=1, j=bunsetsu_cnt）
        if bunsetsu_cnt >= 1:
            cell_text = CkyTable.get_cell_span_text(clauses, 1, bunsetsu_cnt)
        else:
            cell_text = ""
        stage2_candidates = []
        stage2_total = 0
        stage2_parallel = 0
        for ast_entry in stage1_candidates:
            literals = ast_entry.get("literal_list", [])
            # literalが存在する場合は順序通り含まれるかを確認
            ok = True
            if literals:
                pos = 0
                for lit in literals:
                    idx = cell_text.find(lit, pos)
                    if idx == -1:
                        ok = False
                        break
                    pos = idx + len(lit)
            if ok:
                stage2_candidates.append(ast_entry)
                stage2_total += 1
                # ParallelNode含有カウント
                if ast_entry.get("parallel_var_count", 0) >= 2:
                    stage2_parallel += 1
                # 累積カウンタ
                v = ast_entry["var_count"]
                corpus_varcount_stage2[v] += 1
                # Stage2並列含有数
                if ast_entry.get("parallel_var_count", 0) >= 2:
                    corpus_varcount_parallel_stage2[v] += 1

        # Stage3: parallel フィルタ
        # セル内の並列キー数を計算
        if bunsetsu_cnt >= 1:
            key_count = CkyTable.count_parallel_keys(clauses, 1, bunsetsu_cnt, PARALLEL_KEYS)
        else:
            key_count = 0
        stage3_total = 0
        stage3_parallel = 0
        for ast_entry in stage2_candidates:
            par_cnt = ast_entry.get("parallel_var_count", 0)
            # parallel_var_count < 2 の場合は常に通過
            if par_cnt < 2 or key_count >= par_cnt - 1:
                stage3_total += 1
                # ParallelNode含有カウント
                if par_cnt >= 2:
                    stage3_parallel += 1
                # 累積カウンタ
                v = ast_entry["var_count"]
                corpus_varcount_stage3[v] += 1
                if par_cnt >= 2:
                    corpus_varcount_parallel_stage3[v] += 1

        # 文単位ヒストグラム更新
        stage1_hist[stage1_total] += 1
        stage2_hist[stage2_total] += 1
        stage3_hist[stage3_total] += 1

        # Stage1とStage2/Stage3の残り数は、文単位のParallel含有数を計算していないため、
        # Stage1: ast_pool_parallel をstage1_parallelに、ast_pool_nonparallelをstage1_total-stage1_parallelに置き換える
        ast_pool_total = stage1_total
        ast_pool_parallel = stage1_parallel
        ast_pool_nonparallel = stage1_total - stage1_parallel

        # Stage2, Stage3のパス数を記録
        filter_stage1_pass = stage1_total
        filter_stage2_pass = stage2_total
        filter_stage3_pass = stage3_total

        # --- 逐次CSV追記 ---
        rec = {
            "id": sent_id,
            "sentence": sentence,
            "bunsetsu_count": bunsetsu_cnt,
            "ast_pool_total": ast_pool_total,
            "ast_pool_parallel": ast_pool_parallel,
            "ast_pool_nonparallel": ast_pool_nonparallel,
            "by_varcount_json": json.dumps(by_v, ensure_ascii=False, separators=(",", ":")),
            "by_varcount_parallel_json": json.dumps(by_v_parallel, ensure_ascii=False, separators=(",", ":")),
            "filter_stage1_pass": filter_stage1_pass,
            "filter_stage2_pass": filter_stage2_pass,
            "filter_stage3_pass": filter_stage3_pass,
        }
        pd.DataFrame([rec]).to_csv(
            RESULT_STATS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
        )

    elapsed = time.time() - start

    # =========================
    # ▼修正：サマリーJSONに description を併記
    # =========================
    # 各Stage通過AST数のサマリを作成
    summary = {
        "処理文数": {
            "value": total_docs,
            "description": "集計対象となった文の総数"
        },
        "全AST総数(var_count無条件)": {
            "value": dict(sorted(corpus_varcount_total.items())),
            "description": "ASTパターン全体のvar_count別件数（フィルタを一切適用しない状態）"
        },
        "Stage1通過AST総数": {
            "value": dict(sorted(corpus_varcount_stage1.items())),
            "description": "Stage1フィルタ（var_count <= 文節数）を通過したASTのvar_count別件数"
        },
        "Stage1通過AST(並列あり)": {
            "value": dict(sorted(corpus_varcount_parallel_stage1.items())),
            "description": "Stage1通過ASTのうちParallelNodeを含むASTのvar_count別件数"
        },
        "Stage2通過AST総数": {
            "value": dict(sorted(corpus_varcount_stage2.items())),
            "description": "Stage1通過ASTのうちリテラルフィルタを通過したASTのvar_count別件数"
        },
        "Stage2通過AST(並列あり)": {
            "value": dict(sorted(corpus_varcount_parallel_stage2.items())),
            "description": "Stage2通過ASTのうちParallelNodeを含むASTのvar_count別件数"
        },
        "Stage3通過AST総数": {
            "value": dict(sorted(corpus_varcount_stage3.items())),
            "description": "Stage2通過ASTのうち並列数フィルタを通過したASTのvar_count別件数"
        },
        "Stage3通過AST(並列あり)": {
            "value": dict(sorted(corpus_varcount_parallel_stage3.items())),
            "description": "Stage3通過ASTのうちParallelNodeを含むASTのvar_count別件数"
        },
        "文節数ヒストグラム": {
            "value": dict(sorted(bunsetsu_hist.items())),
            "description": "文節数ごとの文の出現数分布（キー=文節数, 値=文数）"
        },
        "文ごとの候補AST総数(Stage1)ヒストグラム": {
            "value": dict(sorted(stage1_hist.items())),
            "description": "各文におけるStage1通過AST件数の分布（キー=AST件数, 値=文数）"
        },
        "文ごとの候補AST総数(Stage2)ヒストグラム": {
            "value": dict(sorted(stage2_hist.items())),
            "description": "各文におけるStage2通過AST件数の分布（キー=AST件数, 値=文数）"
        },
        "文ごとの候補AST総数(Stage3)ヒストグラム": {
            "value": dict(sorted(stage3_hist.items())),
            "description": "各文におけるStage3通過AST件数の分布（キー=AST件数, 値=文数）"
        },
        "メモ": {
            "value": {
                "Stage1": "var_count <= 文節数 で通過",
                "Stage2": "literal フィルタを通過",
                "Stage3": "parallel フィルタを通過"
            },
            "description": "Stageごとのフィルタ条件や備考"
        },
        "出力ファイル": {
            "value": {
                "文別CSV": RESULT_STATS_CSV,
                "サマリJSON": SUMMARY_JSON
            },
            "description": "本処理で出力された統計ファイルのパス"
        },
        "処理時間(秒)": {
            "value": round(elapsed, 2),
            "description": "全処理にかかった時間（秒）"
        }
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"計測時間: {elapsed:.1f} 秒")
    print("=== AST統計の出力完了 ===")
    print("文別CSV:", RESULT_STATS_CSV)
    print("サマリJSON:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
