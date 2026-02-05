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
        # Default guard (enabled) prevents combinatorial blow-ups.
        # Set CKY_GUARD=0 to disable for experiments where maximum recall is preferred.
        guard_off = str(os.getenv("CKY_GUARD", "")).lower() in ("0", "false", "no", "off")

        def _parse_int_with_default(env_key: str, default: int) -> int:
            raw = (os.getenv(env_key, "") or "").strip()
            if raw == "":
                return default
            try:
                v = int(raw)
            except Exception:
                return default
            return v if v > 0 else default

        max_child = None
        max_cell_total = None
        if not guard_off:
            # Defaults per MAIN_PIPELINE_JA.md
            max_child = _parse_int_with_default("CKY_MAX_CHILD_CANDIDATES", 20)
            max_cell_total = _parse_int_with_default("CKY_MAX_CELL_CANDIDATES_TOTAL", 128)

        # Lightweight stats for diagnostics (read by gpu_child_worker/main).
        self.last_stats = {
            "max_child": max_child,
            "max_cell_total": max_cell_total,
            "pair_evals": 0,
            "mask_calls": 0,
            "dep_calls": 0,
            "cells_with_candidates": 0,
            "candidates_total": 0,
        }

        # Search stop condition:
        # The DP parent only consumes up to `max_child` candidates from each child cell.
        # Searching far beyond that can explode model calls without improving downstream results.
        search_target_per_cell = None
        if (max_child is not None) and (max_cell_total is not None):
            search_target_per_cell = min(max_child, max_cell_total)
        elif max_child is not None:
            search_target_per_cell = max_child
        elif max_cell_total is not None:
            search_target_per_cell = max_cell_total

        # Hard safety cap on evaluated (text_A, text_B) pairs per cell to prevent stalls.
        # (Enabled only when guard is ON.)
        max_pair_evals_per_cell = 2048 if not guard_off else None

        # Memoize relation predictions within this sentence/table to reduce repeated calls.
        _pair_cache = {}

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
                pair_evals_in_cell = 0
                for k in range(i, j):
                    left_cell = cky_table[i][k]
                    right_cell = cky_table[k+1][j]
                    if not (isinstance(left_cell, dict) and isinstance(right_cell, dict)):
                        continue
                    left_candidates = (left_cell.get("candidates", []) or [])
                    right_candidates = (right_cell.get("candidates", []) or [])
                    if max_child is not None:
                        left_candidates = left_candidates[:max_child]
                        right_candidates = right_candidates[:max_child]
                    for left_cand in left_candidates:
                        for right_cand in right_candidates:
                            text_A = left_cand.get("text", "")
                            text_B = right_cand.get("text", "")
                            dependency_label = None
                            pred_result = 0
                            acl_result = 0
                            mod_result = 0

                            cache_key = (text_A, text_B)
                            cached = _pair_cache.get(cache_key)
                            if cached is not None:
                                dependency_label, pred_result, acl_result, mod_result = cached
                            else:
                                if self.use_model:
                                    if self.mask_detector is not None:
                                        self.last_stats["mask_calls"] += 1
                                        dependency_label, _ = self.mask_detector.predict_relation(text_A, text_B)
                                        if dependency_label == "項-述語":
                                            pred_result = 1
                                        elif dependency_label == "連体修飾":
                                            acl_result = 1
                                    if (dependency_label is None) and (self.dep_mod_detector is not None):
                                        self.last_stats["dep_calls"] += 1
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
                                _pair_cache[cache_key] = (dependency_label, pred_result, acl_result, mod_result)

                            pair_evals_in_cell += 1
                            self.last_stats["pair_evals"] += 1
                            if (max_pair_evals_per_cell is not None) and (pair_evals_in_cell >= max_pair_evals_per_cell):
                                break


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
                                if (max_cell_total is not None) and (len(candidates) >= max_cell_total):
                                    break
                                if (search_target_per_cell is not None) and (len(candidates) >= search_target_per_cell):
                                    break
                        if (max_cell_total is not None) and (len(candidates) >= max_cell_total):
                            break
                        if (search_target_per_cell is not None) and (len(candidates) >= search_target_per_cell):
                            break
                        if (max_pair_evals_per_cell is not None) and (pair_evals_in_cell >= max_pair_evals_per_cell):
                            break
                    if (max_cell_total is not None) and (len(candidates) >= max_cell_total):
                        break
                    if (search_target_per_cell is not None) and (len(candidates) >= search_target_per_cell):
                        break
                    if (max_pair_evals_per_cell is not None) and (pair_evals_in_cell >= max_pair_evals_per_cell):
                        break
                # セルに候補・全判定結果を格納
                if not isinstance(cky_table[i][j], dict):
                    cky_table[i][j] = {}
                cky_table[i][j]["candidates"] = candidates
                # cky_table[i][j]["all_results"] = all_results   # ← 追加
                if candidates:
                    self.last_stats["cells_with_candidates"] += 1
                    self.last_stats["candidates_total"] += len(candidates)
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
