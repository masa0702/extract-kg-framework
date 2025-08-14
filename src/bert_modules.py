import json

try:
    from mask_module import MaskRelationDetector
    from dep_bert import DependencyModificationRelationDetector
    _BERT_AVAILABLE = True
except Exception:
    # If transformers/torch is not available, fall back to simple heuristics
    MaskRelationDetector = None
    DependencyModificationRelationDetector = None
    _BERT_AVAILABLE = False

from cky_table import CkyTable

class CKYAnalyzer:
    def __init__(self,
                 mask_model_path=None,
                 dep_mod_model_path="../../output_bert_dependency_ver2.0/final_model",
                ):
        if _BERT_AVAILABLE:
            self.mask_detector = MaskRelationDetector(model_name=mask_model_path or "tohoku-nlp/bert-base-japanese-v3")
            self.dep_mod_detector = DependencyModificationRelationDetector(model_path=dep_mod_model_path)
            self.use_model = True
        else:
            # Fallback: no external models
            self.mask_detector = None
            self.dep_mod_detector = None
            self.use_model = False

    def analyze_cky_table(self, cky_table):
        n = len(cky_table) - 1  # CKY表は0行0列がヘッダなので-1
        table_len = len(cky_table)
        # analyze_cky_table 実行前に一次対角セルを全て変換
        for i in range(1, table_len):
            cell = cky_table[i][i]
            if isinstance(cell, dict) and "candidates" not in cell:
                text = cell.get("candidate", "")
                upos = cell.get("upos", [])
                xpos = cell.get("xpos", [])
                cell["candidates"] = [{"text": text, "upos": upos, "xpos": xpos}]

        use_model = self.use_model
        mask_detector = self.mask_detector
        dep_mod_detector = self.dep_mod_detector

        for span in range(2, n + 1):
            for i in range(1, n - span + 2):
                j = i + span - 1
                candidates = []
                for k in range(i, j):
                    left_cell = cky_table[i][k]
                    right_cell = cky_table[k + 1][j]
                    if not (isinstance(left_cell, dict) and isinstance(right_cell, dict)):
                        continue
                    left_candidates = left_cell.get("candidates", [])
                    right_candidates = right_cell.get("candidates", [])
                    if not left_candidates or not right_candidates:
                        continue
                    for left_cand in left_candidates:
                        text_A = left_cand.get("text", "")
                        for right_cand in right_candidates:
                            text_B = right_cand.get("text", "")
                            dependency_label = None
                            if use_model:
                                dependency_label, _ = mask_detector.predict_relation(text_A, text_B)
                                if dependency_label not in ("項-述語", "連体修飾"):
                                    mod_result, _ = dep_mod_detector.predict_relation(text_A, text_B)
                                    if mod_result == 1:
                                        dependency_label = "依存関係"
                            else:
                                # Heuristic fallback: simple rule based on tokens
                                if text_A.endswith("を") or text_B.endswith("する"):
                                    dependency_label = "項-述語"
                                elif text_A.endswith("な") or text_A.endswith("の"):
                                    dependency_label = "連体修飾"

                            if dependency_label is not None:
                                candidates.append({
                                    "left_k": k,
                                    "left": left_cand,
                                    "right": right_cand,
                                    "dependency": {
                                        "daughter_idx": i - 1,
                                        "head_idx": j - 1,
                                        "label": dependency_label
                                    },
                                    "text": text_A + text_B
                                })
                if not isinstance(cky_table[i][j], dict):
                    cky_table[i][j] = {}
                cky_table[i][j]["candidates"] = candidates
        return cky_table


# CkyTable = CkyTable()

# # 利用例
# # input_json = "../data/dependency_analysis.json"  # 先ほど作成したJSONファイル
# # output_json = "../data/dependency_analysis_with_cky.json"

# # # # CKY表を生成して保存
# # CkyTable.process_json_to_cky_and_save(input_json, output_json)

# data_path = "../../data/dependency_analysis_with_cky.json"
# try:
#     with open(data_path, "r", encoding="utf-8") as f:
#         json_data = json.load(f)
# except:
#     pass
# for sentence, data in json_data.items():
#     cky_table = json_data[sentence]["dependency_table"]
#     # CkyTable.display_simple_cky_table(cky_table)
#     # CkyTable.display_multiline_cky_table(cky_table)
    

# analyzer = CKYAnalyzer()
# cky_table_with_dependency = analyzer.analyze_cky_table(cky_table)

# # print(len(cky_table_with_dependency[0]))
# # # 例：結果確認
# for i in range(1, len(cky_table_with_dependency[0])):
#     for j in range(i+1, len(cky_table_with_dependency[0])):
#         cell = cky_table_with_dependency[i][j]
#         k = 0
#         for candidate in cell["candidates"]:
#             print(f"cell[{i}][{j}][candidate {k}] = {candidate}")
#             print()
#             k =+ 1
            

# # CkyTable.display_multiline_cky_table(cky_table)
# # CkyTable.cky_table_to_tsv(cky_table)