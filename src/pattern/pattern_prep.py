# pattern_prep.py
# =============================================================
#  パターン CSV の重複行を削除して上書き保存し、
#  AST 化した結果を Pickle へ保存。
#
#  - 失敗パターンはログにのみ記録
#  - 統計情報を txt で出力
# =============================================================

import pandas as pd
import pickle, gzip, os, traceback
from tqdm.auto import tqdm
from pattern_parser import PatternParser
from pattern_nodes import VariableNode

# ---------- 設定 ----------
INPUT_CSV      = "../data/patterns/all_patterns.csv"
OUTPUT_PICKLE  = "../data/patterns/patterns_ast.pkl.gz"
ERROR_LOG      = "../data/patterns/pattern_prep_errors.log"
STATS_TXT      = "../data/patterns/pattern_prep_stats.txt"
PICKLE_PROTO   = 5  # Python 3.8+

# ---------- 事前ディレクトリ確保 ----------
os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)

# ---------- ① CSV 読み込み & 重複除去 ----------
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
before = len(df)
df_unique = df.drop_duplicates(subset=["pattern"])
after = len(df_unique)
dup_cnt = before - after

# 重複があればファイルを上書き保存
if dup_cnt > 0:
    df_unique.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")

print(f"重複チェック完了: {before} → {after}（{dup_cnt} 行重複）")

# ---------- ② AST 生成 ----------
parser = PatternParser()
succ_cnt = 0
fail_cnt = 0
patterns_ast = []

# エラーログ初期化
open(ERROR_LOG, "w", encoding="utf-8").close()

def count_variables(ast) -> int:
    """AST 内の VariableNode 数を再帰的に数える"""
    cnt = 0
    def visit(node):
        nonlocal cnt
        if isinstance(node, VariableNode):
            cnt += 1
        for attr in ("elements", "options", "block"):
            child = getattr(node, attr, None)
            if not child:
                continue
            if not isinstance(child, list):
                childs = [child]
            else:
                childs = child
            for c in childs:
                visit(c)
    visit(ast)
    return cnt

for idx, pattern in tqdm(enumerate(df_unique["pattern"]),
                         total=after, desc="Parsing patterns"):
    pat = pattern.strip()
    if not pat:
        continue
    try:
        ast = parser.parse(pat)
        patterns_ast.append({
            "pattern": pat,
            "var_count": count_variables(ast),
            "ast": ast
        })
        succ_cnt += 1
    except Exception:
        fail_cnt += 1
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{idx}] {pat}\n{traceback.format_exc()}\n")

# ---------- ③ Pickle 保存 ----------
with gzip.open(OUTPUT_PICKLE, "wb") as fp:
    pickle.dump(patterns_ast, fp, protocol=PICKLE_PROTO)

# ---------- ④ 統計出力 ----------
with open(STATS_TXT, "w", encoding="utf-8") as f:
    f.write(f"パターン総数       : {after}\n")
    f.write(f"AST 生成成功       : {succ_cnt}\n")
    f.write(f"AST 生成失敗       : {fail_cnt}\n")
    f.write(f"重複で削除した行数 : {dup_cnt}\n")
    f.write(f"保存 Pickle        : {OUTPUT_PICKLE}\n")

print("=== 処理完了 ===")
print(f"成功: {succ_cnt} / 失敗: {fail_cnt}")
print(f"Pickle: {OUTPUT_PICKLE}")
print(f"ログ: {ERROR_LOG}")
print(f"統計: {STATS_TXT}")
