import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pattern_nodes import SequenceNode, VariableNode

# 構築した AST の変数数がセルの文節数以下かを確認
ast = SequenceNode([VariableNode("X",1), VariableNode("Y",1)])
var_count = sum(1 for n in ast.walk() if isinstance(n, VariableNode))

chunk_num = 2
assert var_count <= chunk_num
print("bunsetsu filter ok")
