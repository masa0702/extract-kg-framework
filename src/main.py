from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher, MatchResult
import json
import os

# # --- 1) 簡易 CKY 表の準備 -------------------------------------------
# # オリジナルのデータファイルが無いため、サンプル文節から CKY 表を作成する
# clauses = [
#     ["製品情報を", [1, 3], ["製品情報", "を"], ["名詞", "助詞"], [[1, 2], [3, 3]]],
#     ["管理", [4, 5], ["管理"], ["サ変"], [[4, 5]]],
#     ["する", [6, 7], ["する"], ["動詞"], [[6, 7]]],
# ]

# CkyTableObj = CkyTable()
# cky_table = CkyTableObj.create_initializing_cky_table(clauses)

# 1) CKY表の作成
CkyTable = CkyTable()
BASE_DIR = os.path.dirname(__file__)
input_json = os.path.join(BASE_DIR, "..", "data", "dependency_analysis.json")
output_json = os.path.join(BASE_DIR, "..", "data", "dependency_analysis_with_cky.json")

# CKY表を生成して保存
CkyTable.process_json_to_cky_and_save(input_json, output_json)

data_path = output_json

with open(data_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

for sentence, data in json_data.items():
    cky_table = json_data[sentence]["dependency_table"]

# 2) 依存情報付与 (BERT が無い環境でも動作するよう heuristics を使用)
analyzer = CKYAnalyzer()
cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)

# 3) パターンを AST に (簡易サンプル)
pattern = "[*1X1]を[Y1]&[Y2]する"
parser = PatternParser()
ast = parser.parse(pattern)

def post_process(results):
    processed = []
    for r in results:
        m = r.variable_mapping
        x_val = m.get("*1X1", m.get("X1", m.get("X", ""))).rstrip("。").rstrip("を")
        y1 = m.get("Y1")
        y2 = m.get("Y2")
        if y1 is not None and y2 is not None:
            y1_clean = y1.replace("および", "")
            if "する" in y2:
                y1_clean += "する"
            y2_clean = y2.replace("および", "").rstrip("。")
            processed.append(MatchResult(r.cell, r.i, r.j, {"*1X1": x_val, "Y1": y1_clean}))
            processed.append(MatchResult(r.cell, r.i, r.j, {"*1X1": x_val, "Y": y2_clean}))
        else:
            cleaned = {k: v.rstrip("。") for k, v in m.items()}
            processed.append(MatchResult(r.cell, r.i, r.j, cleaned))
    unique = []
    seen = set()
    for r in processed:
        key = tuple(sorted(r.variable_mapping.items()))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# 4) マッチ
matcher = CKYMatcher(ast)
results = matcher.match_table(cky_table_with_dependency)
for r in post_process(results):
    print(f"cell({r.i},{r.j}) ->", r.variable_mapping)
