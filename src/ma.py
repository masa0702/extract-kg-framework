# extract_po_pair_main_fast.py
# =============================================================
# ① AST Pickle をロード
# ② GiNZA 依存解析 + CKY 表キャッシュ
# ③ 文ごとに BERT で CKY解析（メインプロセス）
# ④ ASTマッチングのみ ProcessPoolExecutor で並列化
# ⑤ tqdm で進捗バー表示（2段）
# =============================================================

import os, json, gzip, pickle, time, logging
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
import concurrent.futures as cf
from concurrent.futures import ProcessPoolExecutor
import torch
try:
    import orjson as fastjson  # ==== CHANGED 1 ==== 高速JSON（無ければ標準jsonへフォールバック）
except ImportError:
    fastjson = None

# ---------- 既存モジュール ----------
from pattern_nodes import ParallelNode, VariableNode
from matcher import CKYMatcher
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from clause_analysis import DependencyAnalysis
from utils import MyUtility
from semantic_judge import judge_parallel

# =============================================================
# ログ設定  ==== CHANGED 2 ====
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("extract")

# =============================================================
# 設定
# =============================================================
AST_PICKLE      = "../data/patterns/patterns_ast.pkl.gz"
INPUT_SENT_CSV  = "../data/target_datas/movie_target_data.csv"

dir_name   = os.path.basename(os.path.dirname(INPUT_SENT_CSV))
filename   = os.path.basename(INPUT_SENT_CSV)
prefix     = filename[:-4]
output_dir = f"../results/extract_pred_arg_pair/{dir_name}/{prefix}/"
os.makedirs(output_dir, exist_ok=True)

dep_json_path  = f"{output_dir}{prefix}_dependency_analysis.json"
cky_json_path  = f"{output_dir}{prefix}_dependency_analysis_with_cky.json"
RESULT_CSV     = f"{output_dir}{prefix}_extract_po_pair.csv"

# デバッグ用：長すぎる文で固まる場合のしきい値（秒）  ==== CHANGED 3 ====
WARN_SEC_PER_SENT = 5.0   # 1文あたり5秒超えたら警告
DEBUG_TOP_N        = None # 例: 100 にすると最初の100文だけ処理（Noneで全件）

# =============================================================
# ① AST Pickle 読み込み & dict 化
# =============================================================
print("AST Pickle をロード中…")
with gzip.open(AST_PICKLE, "rb") as fp:
    patterns_ast = pickle.load(fp)

AST_DICT = defaultdict(list)
for entry in patterns_ast:
    AST_DICT[entry["var_count"]].append(entry["ast"])
print(f"ロード完了: {len(patterns_ast)} パターン")

# =============================================================
# ② GiNZA 依存解析 + CKY 表
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CKYAnalyzer デバイス: {device}")

try:
    analyzer = CKYAnalyzer(device=device)
except TypeError:
    analyzer = CKYAnalyzer()
    if device == "cuda" and hasattr(analyzer, "model"):
        analyzer.model.to(device)

CkyTableObj = CkyTable()
depana      = DependencyAnalysis()
myutil      = MyUtility()

sent_df    = pd.read_csv(INPUT_SENT_CSV, dtype=str)
if DEBUG_TOP_N:
    sent_df = sent_df.head(DEBUG_TOP_N)

sentences  = sent_df["sent"].unique().tolist()

# ---- 依存解析キャッシュ ----
try:
    with open(dep_json_path, "r", encoding="utf-8") as f:
        dep_data = json.load(f)
    if not isinstance(dep_data, dict):
        dep_data = {}
except (FileNotFoundError, json.JSONDecodeError):
    dep_data = {}

new_sentences = [s for s in sentences if s not in dep_data]
if new_sentences:
    print(f"GiNZA 解析: {len(new_sentences)} 文")
    dep_results = depana.analyze_sentences(new_sentences)
    dep_data.update(dep_results)
    myutil.save_json_from_file(dep_data, dep_json_path)

# ---- CKY 表キャッシュ ----
if not os.path.exists(cky_json_path):
    print("CKY 表を生成中 …")
    CkyTableObj.process_json_to_cky_and_save(dep_json_path, cky_json_path)

with open(cky_json_path, "r", encoding="utf-8") as f:
    cky_json_data = json.load(f)

print("依存解析 + CKY 準備完了")

# =============================================================
# 共通ヘルパ
# =============================================================
EXCLUDE_POS = ("助詞", "接続詞", "助動詞",
               "補助記号-句点", "補助記号-読点",
               "記号-句点", "記号-読点")

def extract_parallel_variables(ast):
    """ParallelNode 直下の VariableNode のみ取得"""
    vars_ = []
    def visit(node):
        if isinstance(node, ParallelNode):
            vars_.extend([opt for opt in node.options if isinstance(opt, VariableNode)])
        for attr in ("elements", "options", "block"):
            child = getattr(node, attr, None)
            if not child:
                continue
            child_list = child if isinstance(child, list) else [child]
            for c in child_list:
                visit(c)
    visit(ast)
    return [f"{v.symbol}{v.index}" for v in vars_]

def clean_variable_mapping(varmap, clauses):
    new_map = {}
    for var, val in varmap.items():
        found = None
        for cl in clauses:
            if cl[0] == val or (val and cl[0] in val):
                found = cl; break
        if found:
            tokens, xpos = found[2], found[4]
            # any(ex in pos ...) は遅いので startswith に変更できるなら変更推奨
            filtered = [tok for tok, pos in zip(tokens, xpos)
                        if not any(ex in pos for ex in EXCLUDE_POS)]
            if filtered:
                new_map[var] = "".join(filtered)
        else:
            new_map[var] = val
    return new_map

# =============================================================
# ③ メイン側：BERTでCKY解析（計測付き）  ==== CHANGED 4 ====
# =============================================================
def prepare_sentence(row):
    sent_id  = row["id"]
    sentence = row["sent"]

    # キー確認（存在しないと落ちる）
    if sentence not in cky_json_data:
        raise KeyError(f"sentence not found in cky_json_data: {sentence[:30]}")

    info       = cky_json_data[sentence]
    cky_table  = info["dependency_table"]
    clauses    = info["clauses"]
    bun_cnt    = len(cky_table[0])

    # ---- BERT(CKYAnalyzer) 前後で計測＆ログ ----
    start_t = time.time()
    print(f"[CKY-BERT start] id={sent_id}  sent[:20]={sentence[:20]}", flush=True)

    with torch.inference_mode():
        cky_dep = analyzer.analyze_cky_table(cky_table)

    dur = time.time() - start_t
    print(f"[CKY-BERT done ] id={sent_id}  {dur:.2f}s", flush=True)
    if dur > WARN_SEC_PER_SENT:
        print(f"[WARN] slow sentence (> {WARN_SEC_PER_SENT}s): id={sent_id}", flush=True)

    return (sent_id, sentence, cky_dep, clauses, bun_cnt)


# =============================================================
# ③' ワーカー側：ASTマッチングのみ（ProcessPool）
# =============================================================
_AST_DICT = None

def _init_worker(ast_dict):
    """
    プロセス開始時に呼び出され、AST辞書をグローバルに設定。
    """
    global _AST_DICT
    _AST_DICT = ast_dict

def ast_match_worker(args):
    """
    ASTマッチングだけを行い、レコードを返す
    args: (sent_id, sentence, cky_dep, clauses, bunsetsu_cnt)
    """
    sent_id, sentence, cky_dep, clauses, bunsetsu_cnt = args

    # bunsetsu数フィルタで候補ASTを取得
    candidate_asts = []
    for v in range(1, bunsetsu_cnt + 1):
        candidate_asts.extend(_AST_DICT.get(v, []))
    if not candidate_asts:
        return []

    seen = set()
    recs = []
    for ast in candidate_asts:
        matcher = CKYMatcher(ast, verbose=False)
        for r in matcher.match_table(cky_dep):
            key = frozenset(r.variable_mapping.items())
            if key in seen:
                continue
            seen.add(key)

            cmap = clean_variable_mapping(r.variable_mapping, clauses)
            par_names = extract_parallel_variables(ast)
            par_elems = [cmap.get(n) for n in par_names if n in cmap]
            if par_elems and judge_parallel(sentence, par_elems) is False:
                continue

            Xs = [(k, v) for k, v in cmap.items() if k.startswith("X")]
            Ys = [(k, v) for k, v in cmap.items() if k.startswith("Y")]
            if not (Xs and Ys):
                continue

            idx = 0
            for xk, xv in Xs:
                for yk, yv in Ys:
                    recs.append({
                        "id":           sent_id,
                        "sentence":     sentence,
                        "triple_index": idx,
                        "rel_ja":       yv,
                        "arg_ja":       xv
                    })
                    idx += 1
    return recs

# =============================================================
# ④ 実行：CKY(BERT) → ASTマッチング(ProcessPool)
# =============================================================
if __name__ == "__main__":
    # ==== CHANGED 5 ==== BLASやtokenizersの過剰並列を抑制
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("文ごとにCKY解析（BERT）実行中 …")
    prepared = []
    start = time.time()

    # iterrows は遅い → itertuples の方が速いが、ここは可読性優先で現状維持
    for _, row in tqdm(sent_df.iterrows(), total=len(sent_df), desc="CKY(BERT)"):
        prepared.append(prepare_sentence(row))

    bert_elapsed = time.time() - start
    print(f"BERT側完了: {bert_elapsed:.1f} 秒")

    # ---- ProcessPool で AST マッチング ----
    max_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"ASTマッチングを ProcessPool で実行: workers={max_workers}")

    output_records = []
    start = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(AST_DICT,)
    ) as executor:
        futures = [executor.submit(ast_match_worker, task) for task in prepared]
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc="AST match"):
            output_records.extend(f.result())

    elapsed = time.time() - start
    print(f"ASTマッチング処理時間: {elapsed:.1f} 秒")

    # =============================================================
    # ⑤ 結果保存
    # =============================================================
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    print("=== 抽出完了 ===")
    print(f"保存先: {RESULT_CSV}")
    print(f"抽出レコード数: {len(out_df)}")
