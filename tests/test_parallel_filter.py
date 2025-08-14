import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pattern_parser import PatternParser
from pattern_nodes import count_parallel_variables
from cky_table import CkyTable


def parallel_pass(par_cnt, clauses, i, j):
    """parallel フィルタ: 並列キー数が par_cnt-1 以上か"""
    if par_cnt < 2:
        return True
    return CkyTable.count_parallel_keys(clauses, i, j) >= par_cnt - 1


# パターンをパースして AST を構築
parser = PatternParser()
ast = parser.parse("[X1]&[X2]を[Y1]")
par_cnt = count_parallel_variables(ast)
print(par_cnt)

# 並列キーを含む文節リスト（pass）
clauses_ok = [["リンゴと"], ["バナナを"],["食べた"]]
assert parallel_pass(par_cnt, clauses_ok, 1, 3)

# 並列キーを含まない文節リスト（fail）
clauses_ng = [["リンゴは"], ["バナナを"],["食べた"]]
assert not parallel_pass(par_cnt, clauses_ng, 1, 3)

print("parallel filter ok")
