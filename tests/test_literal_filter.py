import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pattern_parser import PatternParser
from pattern_nodes import extract_literal_strings
from cky_table import CkyTable


def literal_pass(literals, clauses, i, j):
    """literal フィルタ: span 内に全リテラルが含まれるか"""
    cell_text = CkyTable.get_cell_span_text(clauses, i, j)
    return all(lit in cell_text for lit in literals)


# パターンをパースして AST を構築
parser = PatternParser()
ast = parser.parse("[X1]を[X2]は[Y1]")
literals = extract_literal_strings(ast)

# テスト用の文節リスト
clauses = [["太郎は"], ["猫を"], ["見た"]]

# セル [1,2] には "猫" が含まれているので通過
assert literal_pass(literals, clauses, 1, 3)

# セル [1,1] には "猫" が含まれないので不通過
# assert not literal_pass(literals, clauses, 1, 3)

print("literal filter ok")
