# matcher.py
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional

from pattern_nodes import PatternNode


# ----------------------------------------------------------------------
#  結果構造
# ----------------------------------------------------------------------
@dataclass
class MatchResult:
    """CKY セル 1 件と，AST 変数 → 文節表層 の対応表"""

    cell: Dict[str, Any]
    i: int  # CKY 表の行 (1-based)
    j: int  # CKY 表の列 (1-based)
    variable_mapping: Dict[str, str]
    node_info: List[tuple] | None = None
    start: int | None = None


# ----------------------------------------------------------------------
#  CKY Matcher 本体
# ----------------------------------------------------------------------
class CKYMatcher:
    """
    span>1 の CKY セル（cell["candidates"] 内の各候補）に対して
    パターン AST を照合し，合致するものを返す。
    """

    # ---------- public ----------
    def __init__(self, pattern_ast: PatternNode) -> None:
        self.pattern_ast = pattern_ast

    def match_table(self, cky_table: List[List[Any]]) -> List[MatchResult]:
        """CKY 表全体を走査して合致セルを返す"""
        matches: List[MatchResult] = []
        n = len(cky_table) - 1
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):  # span > 1
                cell = cky_table[i][j]
                if not isinstance(cell, dict):
                    continue
                for cand in cell.get("candidates", []):
                    res_list = self._match_candidate(cand)
                    if not res_list:
                        continue
                    for varmap, info, start in res_list:
                        matches.append(
                            MatchResult(
                                cell=cand,
                                i=i,
                                j=j,
                                variable_mapping=varmap,
                                node_info=info,
                                start=start,
                            )
                        )
        return matches

    # ---------- internal ----------
    # 3 フェーズ（依存ラベル → リテラル → 品詞）で早期退出
    def _match_candidate(
        self, cand: Dict[str, Any]
    ) -> Optional[List[tuple[Dict[str, str], List[tuple], int]]]:
        if not self._dependency_label_filter(cand):
            return None
        if not self._literal_filter(cand):
            return None
        infos = self._pos_and_variable_filter(cand)
        if infos is None:
            return None
        results = []
        for info, start in infos:
            varmap = {
                ident: "".join(tokens)
                for ident, tokens, kind, pos, _ in info
                if kind in ("modifier", "variable")
            }
            results.append((varmap, info, start))
        return results

    # --- phase-1 : 依存ラベル本数 ---
    def _dependency_label_filter(self, cand: Dict[str, Any]) -> bool:
        required: Dict[str, int] = (
            self.pattern_ast.get_dependency_label_requirements() or {}
        )
        if not required:
            return True
        actual_counter = Counter(self._collect_dep_labels(cand))
        for label, need in required.items():
            if actual_counter.get(label, 0) < need:
                return False
        return True

    # --- phase-2 : リテラル ---
    def _literal_filter(self, cand: Dict[str, Any]) -> bool:
        literal_nodes = self.pattern_ast.get_literal_nodes()
        if not literal_nodes:
            return True
        cand_text = cand.get("text", "")
        for tokens, _ in literal_nodes:
            lit = "".join(tokens)
            if lit not in cand_text:
                return False
        return True

    # --- phase-3 : 品詞 + 変数割当 ---
    def _pos_and_variable_filter(
        self, cand: Dict[str, Any]
    ) -> Optional[List[tuple[list[tuple], int]]]:
        leaves = self._collect_leaves(cand)
        if not leaves:
            return None

        node_info = self.pattern_ast.get_node_span_info()
        total_span = sum(
            span for _, kind, span, _ in node_info if kind in ("modifier", "variable")
        )
        literal_nodes = ["".join(t) for t, _ in self.pattern_ast.get_literal_nodes()]

        def pos_match(pos_set: set[str], toks: list[str], tag: str) -> bool:
            if tag == "名詞":
                return "NOUN" in pos_set
            if tag == "動詞":
                return "VERB" in pos_set or "AUX" in pos_set
            if tag.startswith("サ変"):
                if "する" in toks:
                    return True
                return "NOUN" in pos_set or "VERB" in pos_set
            return True

        matches: List[tuple[list[tuple], int]] = []
        for start in range(len(leaves)):
            sub_text = "".join(
                "".join(
                    leaf.get("tokens")
                    or [leaf.get("candidate") or leaf.get("text", "")]
                )
                for leaf in leaves[start:]
            )
            if any(lit not in sub_text for lit in literal_nodes):
                continue
            idx = start
            results: List[tuple] = []  # (ident, tokens, kind, pos_tag, pos_set)
            ok = True
            for ident, kind, span_len, pos_tag in node_info:
                tokens: List[str] = []
                pos_set: set[str] = set()
                if kind in ("modifier", "variable"):
                    if idx + span_len > len(leaves):
                        ok = False
                        break
                    for _ in range(span_len):
                        leaf = leaves[idx]
                        idx += 1
                        tokens.extend(
                            leaf.get("tokens")
                            or [leaf.get("candidate") or leaf.get("text", "")]
                        )
                        pos_set.update(leaf.get("pos", []))
                    if kind == "variable" and pos_tag:
                        if not pos_match(pos_set, tokens, pos_tag):
                            ok = False
                            break
                results.append((ident, tokens, kind, pos_tag, pos_set))
            if ok and idx <= len(leaves):
                matches.append((results, start))
        return matches or None

    # ---------- utility ----------
    def _collect_dep_labels(self, node: Dict[str, Any]) -> List[str]:
        """候補部分木を DFS して依存ラベルを列挙"""
        labels = []
        if "dependency" in node and isinstance(node["dependency"], dict):
            lbl = node["dependency"].get("label")
            if lbl:
                labels.append(lbl)
        for side in ("left", "right"):
            if side in node and isinstance(node[side], dict):
                labels.extend(self._collect_dep_labels(node[side]))
        return labels

    def _collect_leaves(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """候補部分木の葉（対角線セル辞書）を左→右順で取得"""
        if "left" in node and "right" in node:
            return self._collect_leaves(node["left"]) + self._collect_leaves(
                node["right"]
            )
        return [node]  # leaf
