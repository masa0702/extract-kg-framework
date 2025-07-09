import pandas as pd
import json
import os

from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher
from clause_analysis import DependencyAnalysis
from utils import MyUtility
from semantic_judge import judge_parallel  # True/False判定用

# -- 準備 --
CkyTableObj = CkyTable()
analyzer = CKYAnalyzer()
parser = PatternParser()
depana = DependencyAnalysis()
myutil = MyUtility()

# --- CSV読み込み ---
INPUT_CSV = "../data/fix_parallel_subject-object_ja_sent_po_pair/fix_1_movie_ontology_parallel_subject-object_ja_sent_po_pair.csv"
df = pd.read_csv(INPUT_CSV, dtype=str)

# ディレクトリ名（最後の一つだけ）
dir_name = os.path.basename(os.path.dirname(INPUT_CSV))
# ファイル名
filename = os.path.basename(INPUT_CSV)

# ファイル名の接頭辞（"1_movie_ontology_"）
if filename.endswith(f"{dir_name}.csv"):
    prefix = filename[:-(len(dir_name)+4)]  # ".csv"も4文字なので+4
else:
    prefix = filename  # 万一一致しなければ全体

# 結果
# print("ディレクトリ名:", dir_name)
# print("ファイル名プレフィックス:", prefix)

# --- 依存構造解析とCKY表作成 ---
sentences = df['sent_ja'].unique().tolist()
output_dir = f"../results/extract_po_pair/{dir_name}/{prefix}/"
os.makedirs(output_dir, exist_ok=True) 
output_filename = f"{output_dir}{prefix}dependency_analysis.json"

try:
    with open(output_filename, "r", encoding="utf-8") as f:
        dep_data = json.load(f)
    if not isinstance(dep_data, dict):
        dep_data = {}
except (FileNotFoundError, json.JSONDecodeError):
    dep_data = {}

new_sentences = [s for s in sentences if s not in dep_data]
if new_sentences:
    new_results = depana.analyze_sentences(new_sentences)
    dep_data.update(new_results)
    myutil.save_json_from_file(dep_data, output_filename)

input_json = output_filename
output_json = f"{output_dir}{prefix}dependency_analysis_with_cky.json"

CkyTableObj.process_json_to_cky_and_save(input_json, output_json)

with open(output_json, "r", encoding="utf-8") as f:
    cky_json_data = json.load(f)

print("CKY表初期化完了")
# --- 関数: ParallelNode直下のVariableNode名リスト ---
def extract_parallel_variables(ast):
    result = []
    from pattern_nodes import ParallelNode, VariableNode
    def visit(node):
        if isinstance(node, ParallelNode):
            for opt in node.options:
                if isinstance(opt, VariableNode):
                    result.append(opt)
        if hasattr(node, "elements"):
            for child in node.elements:
                visit(child)
        if hasattr(node, "block"):
            visit(node.block)
    visit(ast)
    return result

# --- 変数値クリーニング ---
EXCLUDE_POS = [
    "助詞", "接続詞", "助動詞",
    "補助記号-句点", "補助記号-読点",
    "記号-句点", "記号-読点"
]
def clean_variable_mapping(varmap, clauses):
    new_varmap = {}
    for var, val in varmap.items():
        found = None
        for clause in clauses:
            surface = clause[0]
            if surface == val or (val and surface in val):
                found = clause
                break
        if found:
            tokens = found[2]
            xpos = found[4]
            filtered = [tok for tok, pos in zip(tokens, xpos)
                        if not any(ex in pos for ex in EXCLUDE_POS)]
            new_val = "".join(filtered)
            if new_val:
                new_varmap[var] = new_val
            else:
                continue
        else:
            new_varmap[var] = val
    return new_varmap

# --- メイン処理 ---
output_results = []

for idx, row in df.iterrows():
    id = row["id"]
    sentence = row['sent_ja']
    ont_id = row['ont_id']
    patterns = [p.strip() for p in str(row['pattern']).split(',')]
    cky_table = cky_json_data[sentence]["dependency_table"]
    clauses = cky_json_data[sentence]["clauses"]
    cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)
    # print(patterns)
    # exit()
    for pattern in patterns:
        print(sentence, pattern)
        ast = parser.parse(pattern)
        ast.debug()
        parallel_vars = extract_parallel_variables(ast)
        parallel_var_names = [f"{v.symbol}{v.index}" for v in parallel_vars]
        matcher = CKYMatcher(ast, verbose=True)
        results = matcher.match_table(cky_table_with_dependency)
        seen = []
        # print("gjkladsjgsajgja", results)
        for r in results:
            if r.variable_mapping in seen:
                continue
            seen.append(r.variable_mapping)
            new_varmap = clean_variable_mapping(r.variable_mapping, clauses)
            # print(r.variable_mapping)

            updated_parallel_elements_list = [new_varmap.get(var_name) for var_name in parallel_var_names if var_name in new_varmap]
            
            # 1. 並列要素が空も含めて保存対象とする
            # 2. semantic_judgeでFalseならスキップ
            if updated_parallel_elements_list:
                judge_result = judge_parallel(sentence, updated_parallel_elements_list)
                print(judge_result)
                if judge_result is False:
                    continue  # Falseは保存しない
            # 空リストはjudge_parallelせず、そのまま保存

            # 3. new_varmapのXn, Ynの組み合わせで全件保存
            X_vars = sorted([(k, v) for k, v in new_varmap.items() if k.startswith('X')])
            Y_vars = sorted([(k, v) for k, v in new_varmap.items() if k.startswith('Y')])
            triple_index = 0
            for xk, xv in X_vars:
                for yk, yv in Y_vars:
                    output_results.append({
                        "id": id,
                        "ont_id": ont_id,
                        "sentence": sentence,
                        "triple_index": triple_index,
                        "sub_ja": None,
                        "rel_ja": yv,
                        "obj_ja": xv
                    })
                    triple_index += 1
            print(f"matchi end {pattern}")
            # print("updated_parallel_elements_list:", updated_parallel_elements_list)
            # print("judge_result:", judge_result if updated_parallel_elements_list else "N/A")

# --- 結果保存例 ---
out_df = pd.DataFrame(output_results)
save_results_path = f"{output_dir}{prefix}extract_po_pair.csv"
out_df.to_csv(save_results_path, index=False, encoding="utf-8-sig")
print(f"保存しました: {save_results_path}")
