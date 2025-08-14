import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pattern_nodes import SequenceNode, VariableNode, ParallelNode, count_parallel_variables
from cky_table import CkyTable

clauses = [
    ["AとBや"],
    ["C"],
]

ast = SequenceNode([ParallelNode([VariableNode("X",1), VariableNode("X",2)]), VariableNode("Y",1)])
par_cnt = count_parallel_variables(ast)
cell_parallel = CkyTable.count_parallel_keys(clauses, 1, 1)
assert cell_parallel >= par_cnt - 1
print("parallel filter ok")
