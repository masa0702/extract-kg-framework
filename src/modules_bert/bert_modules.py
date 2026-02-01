import json
import os

try:
    from .mask_module import MaskRelationDetector
    from .dep_bert import DependencyModificationRelationDetector
    _BERT_AVAILABLE = True
except Exception:
    # If transformers/torch is not available, fall back to simple heuristics
    MaskRelationDetector = None
    DependencyModificationRelationDetector = None
    _BERT_AVAILABLE = False

from modules_core.cky_table import CkyTable

class CKYAnalyzer:
    def __init__(self,
                 mask_model_path=None,
                 dep_mod_model_path=None,
                ):
        self.mask_detector = None
        self.dep_mod_detector = None
        self.use_model = False
        self.model_error = ""

        if _BERT_AVAILABLE:
            disable_mask = str(os.getenv("DISABLE_MASK_BERT", "")).lower() in ("1", "true", "yes")
            env_mask_path = os.getenv("MASK_BERT_MODEL_PATH", "").strip()
            env_mask_name = os.getenv("MASK_BERT_MODEL_NAME", "").strip()
            if env_mask_path:
                mask_model_path = env_mask_path
            try:
                if not disable_mask:
                    self.mask_detector = MaskRelationDetector(
                        model_name=env_mask_name or mask_model_path or "tohoku-nlp/bert-base-japanese-v3"
                    )
            except Exception as e:
                self.model_error = f"mask_detector init failed: {e}"
                self.mask_detector = None

            if dep_mod_model_path is None:
                dep_mod_model_path = os.getenv(
                    "DEP_BERT_MODEL_PATH",
                    "/workspace/src/modules_bert/models/output_bert_dependency_bunsetsu_ver3.0/depbert_bunsetsu_20260117_072956/final_model",
                )
            try:
                if not dep_mod_model_path or not os.path.isdir(dep_mod_model_path):
                    raise FileNotFoundError(f"dep_bert model not found: {dep_mod_model_path}")
                self.dep_mod_detector = DependencyModificationRelationDetector(
                    model_path=dep_mod_model_path
                )
            except Exception as e:
                if self.model_error:
                    self.model_error += " | "
                self.model_error += f"dep_mod_detector init failed: {e}"
                self.dep_mod_detector = None

            if self.dep_mod_detector is None:
                raise RuntimeError(self.model_error or "dep_mod_detector init failed")

            if (self.mask_detector is not None) or (self.dep_mod_detector is not None):
                self.use_model = True
        else:
            self.use_model = False

    def analyze_cky_table(self, cky_table):
        n = len(cky_table) - 1  # CKY表は0行0列がヘッダなので-1
        # NOTE:
        # Building candidates by enumerating every combination of (left_candidates x right_candidates)
        # across spans can explode combinatorially and make even 1 sentence take >10 minutes.
        # We cap how many child candidates are considered when combining spans.
        try:
            max_child = int(os.getenv("CKY_MAX_CHILD_CANDIDATES", "1"))
        except Exception:
            max_child = 1
        try:
            max_cell_total = int(os.getenv("CKY_MAX_CELL_CANDIDATES_TOTAL", "64"))
        except Exception:
            max_cell_total = 64
        max_child = max(1, max_child)
        max_cell_total = max(1, max_cell_total)

        # analyze_cky_table 実行前に一次対角セルを全て変換
        for i in range(1, len(cky_table)):
            cell = cky_table[i][i]
            if isinstance(cell, dict) and "candidates" not in cell:
                text = cell.get("candidate", "")
                upos = cell.get("upos", [])
                xpos = cell.get("xpos", [])
                cell["candidates"] = [{"text": text, "upos": upos, "xpos": xpos}]

        for span in range(2, n+1):
            for i in range(1, n-span+2):
                j = i + span - 1
                candidates = []
                all_results = []  # ← 全てのペアと判定結果を記録
                for k in range(i, j):
                    left_cell = cky_table[i][k]
                    right_cell = cky_table[k+1][j]
                    if not (isinstance(left_cell, dict) and isinstance(right_cell, dict)):
                        continue
                    left_candidates = (left_cell.get("candidates", []) or [])[:max_child]
                    right_candidates = (right_cell.get("candidates", []) or [])[:max_child]
                    for left_cand in left_candidates:
                        for right_cand in right_candidates:
                            text_A = left_cand.get("text", "")
                            text_B = right_cand.get("text", "")
                            dependency_label = None
                            pred_result = 0
                            acl_result = 0
                            mod_result = 0

                            if self.use_model:
                                if self.mask_detector is not None:
                                    dependency_label, _ = self.mask_detector.predict_relation(text_A, text_B)
                                    if dependency_label == "項-述語":
                                        pred_result = 1
                                    elif dependency_label == "連体修飾":
                                        acl_result = 1
                                if (dependency_label is None) and (self.dep_mod_detector is not None):
                                    mod_result, _ = self.dep_mod_detector.predict_relation(text_A, text_B)
                                    if mod_result == 1:
                                        dependency_label = "依存関係"
                            else:
                                # Heuristic fallback: simple rule based on tokens
                                if text_A.endswith("を") or text_B.endswith("する"):
                                    dependency_label = "項-述語"
                                    pred_result = 1
                                elif text_A.endswith("な") or text_A.endswith("の"):
                                    dependency_label = "連体修飾"
                                    acl_result = 1


                            # 追加: すべてのペアと判定結果を保存
                            all_results.append({
                                "left": text_A,
                                "right": text_B,
                                "pred_result": pred_result,
                                "acl_result": acl_result,
                                "mod_result": mod_result,
                                "dependency_label": dependency_label
                            })

                            # 従来どおり候補としても保存
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
                                if len(candidates) >= max_cell_total:
                                    break
                        if len(candidates) >= max_cell_total:
                            break
                    if len(candidates) >= max_cell_total:
                        break
                # セルに候補・全判定結果を格納
                if not isinstance(cky_table[i][j], dict):
                    cky_table[i][j] = {}
                cky_table[i][j]["candidates"] = candidates
                # cky_table[i][j]["all_results"] = all_results   # ← 追加
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
