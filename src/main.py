from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher
from clause_analysis import DependencyAnalysis
from utils import MyUtility
import json

# --- 1) 簡易 CKY 表の準備 -------------------------------------------
# オリジナルのデータファイルが無いため、サンプル文節から CKY 表を作成する
# clauses = [
#     ["製品情報を", [1, 3], ["製品情報", "を"], ["名詞", "助詞"], [[1, 2], [3, 3]]],
#     ["管理", [4, 5], ["管理"], ["サ変"], [[4, 5]]],
#     ["する", [6, 7], ["する"], ["動詞"], [[6, 7]]],
# ]

sentences = [
        # "技術的な製品情報を記述および管理する。",
        # "技術的な製品情報を記述および管理、準備する。",
        # "ジヴコ・スリェプチェヴィッチは、トレリサックFCのスポーツチームのメンバーであり、コーチでもあります。",
        # "製品情報と案件情報の書類を管理および整理する。",
        "上司の重要な仕事と案件を整理する"
    ]
depana = DependencyAnalysis()
output_filename = "../data/dependency_analysis.json"
try:
    with open(output_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

new_sentences = [s for s in sentences if s not in data]
myutil = MyUtility()
if new_sentences:
    new_results = depana.analyze_sentences(new_sentences)
    data.update(new_results)
    myutil.save_json_from_file(data, output_filename)


input_json = "../data/dependency_analysis.json"  # 先ほど作成したJSONファイル
output_json = "../data/dependency_analysis_with_cky.json"

CkyTableObj = CkyTable()
CkyTableObj.process_json_to_cky_and_save(input_json, output_json)

try:
    with open(output_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)
except:
    pass
for sentence, data in json_data.items():
    cky_table = json_data[sentence]["dependency_table"]
# 2) 依存情報付与 (BERT が無い環境でも動作するよう heuristics を使用)
analyzer = CKYAnalyzer()
cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)
CkyTableObj.display_multiline_cky_table(cky_table_with_dependency)
# 3) パターンを AST に (簡易サンプル)
# pattern = "[*1X1-名詞]を[Y1-サ変]&[Y2-サ変]する"
# pattern = "[X1-名詞]を[Y1-サ変]&[Y2-サ変]&[Y3-サ変]する"
# pattern = "[*([M1]&[M2])X1-名詞]を[Y1-サ変]&[Y2-サ変]する"
pattern = "[X1]の[*1([Y1-名詞]&[Y2-名詞])]"
parser = PatternParser()
ast = parser.parse(pattern)
ast.debug()

# 4) マッチ
matcher = CKYMatcher(ast)
results = matcher.match_table(cky_table_with_dependency)
# print(cky_table_with_dependency[1][4])
for r in results:
    print(f"cell({r.i},{r.j}) ->", r.variable_mapping)
ast.debug()