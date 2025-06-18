import json

from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher

# 1) CKY表の作成
CkyTable = CkyTable()
input_json = "../data/dependency_analysis.json"
output_json = "../data/dependency_analysis_with_cky.json"

# CKY表を生成して保存
CkyTable.process_json_to_cky_and_save(input_json, output_json)

data_path = "../data/dependency_analysis_with_cky.json"

with open(data_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

for sentence, data in json_data.items():
    cky_table = json_data[sentence]["dependency_table"]

# 2) BERT で依存情報付与
analyzer = CKYAnalyzer()
cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)

# 3) パターンを AST に
pattern = "[*1X1-名詞]を[Y1-サ変]&[Y2-サ変]する"
parser = PatternParser()
ast = parser.parse(pattern)
ast.debug()

# 4) マッチ
matcher = CKYMatcher(ast)
results = matcher.match_table(cky_table)
for r in results:
    print(f"cell({r.i},{r.j}) ->", r.variable_mapping)
