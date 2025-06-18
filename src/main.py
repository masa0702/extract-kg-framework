from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher

# --- 1) 簡易 CKY 表の準備 -------------------------------------------
# オリジナルのデータファイルが無いため、サンプル文節から CKY 表を作成する
clauses = [
    ["製品情報を", [1, 3], ["製品情報", "を"], ["名詞", "助詞"], [[1, 2], [3, 3]]],
    ["管理", [4, 5], ["管理"], ["サ変"], [[4, 5]]],
    ["する", [6, 7], ["する"], ["動詞"], [[6, 7]]],
]

CkyTableObj = CkyTable()
cky_table = CkyTableObj.create_initializing_cky_table(clauses)

# 2) 依存情報付与 (BERT が無い環境でも動作するよう heuristics を使用)
analyzer = CKYAnalyzer()
cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)

# 3) パターンを AST に (簡易サンプル)
pattern = "[X1-名詞]を[Y1-サ変]する"
parser = PatternParser()
ast = parser.parse(pattern)

# 4) マッチ
matcher = CKYMatcher(ast)
results = matcher.match_table(cky_table_with_dependency)
for r in results:
    print(f"cell({r.i},{r.j}) ->", r.variable_mapping)
