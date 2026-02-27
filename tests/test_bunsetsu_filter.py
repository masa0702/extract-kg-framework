import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pattern.pattern_parser import PatternParser
from pattern.pattern_nodes import VariableNode
from modules_core.cky_table import CkyTable


def bunsetsu_pass(ast, i, j):
    """bunsetsu フィルタ: 変数数がセル文節数以下なら通過"""
    var_count = sum(1 for n in ast.walk() if isinstance(n, VariableNode))
    chunk_num = j - i + 1
    return var_count <= chunk_num


# パターンをパースして AST を構築
parser = PatternParser()
ast = parser.parse("[X1]は[Y2]")

# テスト用の文節リスト（2文節）
clauses = [["太郎は"], ["走った"]]

# セル [1,2] の文を取得して確認
assert CkyTable.get_cell_span_text(clauses, 1, 2) == "太郎は走った"

# セル [1,2] は 2 文節なのでフィルタ通過
assert bunsetsu_pass(ast, 1, 2)

# セル [1,1] は 1 文節なのでフィルタ不通過
assert not bunsetsu_pass(ast, 1, 1)

print("bunsetsu filter ok")
