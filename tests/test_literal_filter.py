import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pattern_nodes import SequenceNode, LiteralNode, VariableNode, extract_literal_strings
from cky_table import CkyTable

# ダミーの文節データ
clauses = [
    ["私は"],
    ["リンゴを食べた"],
]

ast = SequenceNode([LiteralNode(["リンゴ"]), VariableNode("X",1)])
lits = extract_literal_strings(ast)
cell_text = CkyTable.get_cell_span_text(clauses, 1, 2)
assert all(l in cell_text for l in lits)
print("literal filter ok")
