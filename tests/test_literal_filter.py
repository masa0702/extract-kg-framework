import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pattern_parser import PatternParser
from pattern_nodes import extract_literal_strings
from cky_table import CkyTable


def literal_pass(literals, clauses, i, j):
    """literal フィルタ: span 内にリテラルが順序通り含まれるか"""
    cell_text = CkyTable.get_cell_span_text(clauses, i, j)
    pos = 0
    for lit in literals:
        idx = cell_text.find(lit, pos)
        if idx == -1:
            return False
        pos = idx + len(lit)
    return True


# パターンをパースして AST を構築
parser = PatternParser()
ast = parser.parse("[X1]猫[Y2]犬")
literals = extract_literal_strings(ast)

# 正しい順序でリテラルが現れる文節リスト
clauses_ok = [["猫"], ["が"], ["犬"]]
assert literal_pass(literals, clauses_ok, 1, 3)

# 2つ目のリテラルが含まれない範囲では不通過
assert not literal_pass(literals, clauses_ok, 1, 2)

# リテラルの順序が逆の場合も不通過
clauses_rev = [["犬"], ["が"], ["猫"]]
assert not literal_pass(literals, clauses_rev, 1, 3)

print("literal filter ok")
