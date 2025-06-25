# matcher.py
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from pattern_nodes import (
    PatternNode,
    VariableNode,
    ModifierSingleNode,
    ModifierParallelNode,
    ModifierRepeatNode,
    ModifierBlockRepeatNode,
    LiteralNode,
)


# ----------------------------------------------------------------------
#  結果構造
# ----------------------------------------------------------------------
@dataclass
class MatchResult:
    """CKY セル 1 件と，AST 変数 → 文節表層 の対応表"""
    cell: Dict[str, Any]
    i: int              # CKY 表の行 (1-based)
    j: int              # CKY 表の列 (1-based)
    variable_mapping: Dict[str, str]


# ----------------------------------------------------------------------
#  CKY Matcher 本体
# ----------------------------------------------------------------------
class CKYMatcher:
    """
    span>1 の CKY セル（cell["candidates"] 内の各候補）に対して
    パターン AST を照合し，合致するものを返す。
    """

    def __init__(self, pattern_ast: PatternNode) -> None:
        self.pattern_ast = pattern_ast

    # ---------- public ----------
    def match_table(self, cky_table: List[List[Any]]) -> List[MatchResult]:
        """CKY 表全体を走査して合致セルを返す"""
        matches: List[MatchResult] = []
        n = len(cky_table) - 1
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):  # span > 1
                cell = cky_table[i][j]
                if not isinstance(cell, dict):
                    continue

                for cand_idx, cand in enumerate(cell.get("candidates", []), start=1):
                    print(f"\n=== CKY({i},{j}) 候補 {cand_idx} : '{cand.get('text', '')[:30]}...' ===")

                    res = self._match_candidate(cand)
                    if res:
                        print("✅ 最終的にマッチしました")
                        matches.append(MatchResult(
                            cell=cand, i=i, j=j, variable_mapping=res
                        ))
                    else:
                        print("❌ マッチしませんでした")
        print(f"\n==>  合致セル総数: {len(matches)}")
        return matches

    # ---------- internal ----------
    # 3 フェーズ（依存ラベル → リテラル → 品詞）で早期退出
    def _match_candidate(self, cand: Dict[str, Any]) -> Optional[Dict[str, str]]:
        # 0-A. DFS 出現順 ID を付与
        self._assign_seq_ids()                 # ★ 追加

        # 0-B. 葉インデックスを割り振る
        leaves = self._collect_leaves(cand)
        ok, _, _ = self._assign_dynamic_indices(self.pattern_ast, leaves)
        if not ok:
            print("  [prep] AST への動的インデックス付与に失敗")
            return None

        # 1) 依存ラベル
        if not self._dependency_label_filter(cand):
            print("  [phase-1] 依存ラベル要件を満たさず失敗")
            return None
        print("  [phase-1] 依存ラベル要件をパス")

        # 2) リテラル
        if not self._literal_filter(cand):
            print("  [phase-2] リテラル不一致で失敗")
            return None
        print("  [phase-2] リテラル一致でパス")

        # 3) 品詞+変数
        varmap = self._pos_and_variable_filter(cand)
        if varmap is None:
            print("  [phase-3] 品詞チェック／変数割当で失敗")
        else:
            print(f"  [phase-3] 品詞チェックをパス → varmap={varmap}")
        return varmap


    # --- phase-1 : 依存ラベル本数 ---
    def _dependency_label_filter(self, cand: Dict[str, Any]) -> bool:
        required: Dict[str, int] = (
            self.pattern_ast.get_dependency_label_requirements() or {}
        )
        if not required:
            return True
        actual_counter = Counter(self._collect_dep_labels(cand))
        for label, need in required.items():
            got = actual_counter.get(label, 0)
            print(f"    ├─ label='{label}': need={need}, got={got}")
            if got < need:
                return False
        return True


    # --- phase-2 : リテラル ---
    def _literal_filter(self, cand: Dict[str, Any]) -> bool:
        literal_nodes = self.pattern_ast.get_literal_nodes()  # [(tokens, leaf_idx), ...]
        if not literal_nodes:
            return True

        leaves = self._collect_leaves(cand)
        total_leaves = len(leaves)

        # デバッグ: 葉の中身確認
        # for k, leaf in enumerate(leaves):
        #     print(f"      [leaf {k}] {self._collect_text_recursive(leaf)}")

        for tokens, leaf_idx in literal_nodes:
            lit = "".join(tokens)

            # (A) 位置情報なし → 従来どおり
            if leaf_idx is None:
                if lit not in cand.get("text", ""):
                    print(f"    ├─ リテラル '{lit}' が cand_text に未出現 (leaf_idx=None)")
                    return False
                continue

            # (B) leaf_idx が int / list[int] （**0-based**）
            idx_list = (
                [leaf_idx] if isinstance(leaf_idx, int)
                else leaf_idx if isinstance(leaf_idx, list)
                else None
            )
            if idx_list is None:
                print(f"    ├─ leaf_idx の型が想定外: {leaf_idx!r}")
                return False

            # 0-based 境界チェック
            if any(idx < 0 or idx >= total_leaves for idx in idx_list):
                print(f"    ├─ リテラル '{lit}': leaf_idx {idx_list} が範囲外 (葉数={total_leaves})")
                return False

            # 指定 leaf 群について再帰連結テキストを生成
            concat_text = "".join(
                self._collect_text_recursive(leaves[idx]) for idx in idx_list
            )
            if lit not in concat_text:
                print(f"    ├─ リテラル '{lit}' が leaf{idx_list}='{concat_text}' 内に未出現")
                return False

        return True


    # --- phase-3 : 品詞 + 変数割当 ---
    def _pos_and_variable_filter(self, cand: Dict[str, Any]) -> Optional[Dict[str, str]]:
        leaves = self._collect_leaves(cand)
        if not leaves:
            return None

        var_constraints = self.pattern_ast.get_variable_constraints()  # [(sym, idx, pos_tag)]
        varmap: Dict[str, str] = {}
        counter = defaultdict(int)   # ← 追加: 各シンボルの出現回数

        for sym, leaf_idx, pos_tag in var_constraints:   # ← idx → leaf_idx
            if leaf_idx >= len(leaves):
                print(f"    ├─ 変数{sym}: leaf_idx={leaf_idx} が葉数={len(leaves)} を超過")
                return None

            leaf = leaves[leaf_idx]
            surface = leaf.get("candidate") or leaf.get("text", "")

            # ---------- 品詞チェック ----------
            if pos_tag:
                raw_pos = leaf.get("xpos") or leaf.get("pos") or leaf.get("upos") or []
                flat_pos = []
                for p in raw_pos:
                    flat_pos.extend(p if isinstance(p, (list, tuple)) else [p])

                if not any(pos_tag in p for p in flat_pos):
                    print(f"    ├─ 変数{sym}: pos_tag='{pos_tag}' 未一致 (leaf_pos={flat_pos})")
                    return None

            # ---------- 変数名生成 ----------
            counter[sym] += 1
            key = f"{sym}{counter[sym]}"   # 例: Y → Y1, Y2 …
            varmap[key] = surface
            print(f"    ├─ 変数{key}: '{surface}' を割当 (pos OK)")

        return varmap


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
            return self._collect_leaves(node["left"]) + self._collect_leaves(node["right"])
        return [node]  # leaf
    
    def _collect_text_recursive(self, node: Dict[str, Any]) -> str:
        """ノード以下にある 'text' を left→right 順で連結して返す"""
        if node is None or not isinstance(node, dict):
            return ""
        if "left" in node and "right" in node:
            return self._collect_text_recursive(node["left"]) + self._collect_text_recursive(node["right"])
        return node.get("text", "")
    
    # 直後にリテラルがあるかを先読み
    def _next_literal_tokens(self, parent_children: list, cur_idx: int) -> Optional[str]:
        if cur_idx + 1 < len(parent_children):
            nxt = parent_children[cur_idx + 1]
            if isinstance(nxt, LiteralNode):
                return "".join(nxt.text_tokens)
        return None


    # ------------------------------------------------------------------
    #  パターン AST を DFS 前順で走査し seq_id を付与
    # ------------------------------------------------------------------
    def _assign_seq_ids(self) -> None:
        cur = 1
        for node in self.pattern_ast.walk():

            # Variable / 単発・並列修飾
            if isinstance(node, (VariableNode, ModifierSingleNode, ModifierParallelNode)):
                node.seq_id = cur
                cur += 1

            # 繰返し修飾 *n / #n
            elif isinstance(node, ModifierRepeatNode):
                node.seq_id = cur
                cur += node.count                    # n 個飛ばす

            # ブロック修飾は番号を振らない
            elif isinstance(node, ModifierBlockRepeatNode):
                node.seq_id = None

            # Literal は 1 つ前のノードに含まれる扱い
            elif isinstance(node, LiteralNode):
                node.seq_id = None

            # その他は番号なし
            else:
                node.seq_id = None


    # ------------------------------------------------------------------
    #  AST へ “その候補の葉順” に合わせて leaf_idx を付与 & ログ出力
    # ------------------------------------------------------------------
    def _assign_dynamic_indices(
        self,
        node: PatternNode,
        leaves: List[Dict[str, Any]],
        leaf_ptr: int = 0,
        counters: Optional[defaultdict] = None,
        last_var_idx: Optional[int] = None,
        parent_children: Optional[list] = None,
        idx_in_parent: int = -1,
        depth: int = 0,                       # ← ★ 追加：表示用インデント
    ) -> tuple[bool, int, Optional[int]]:
        """
        戻り値: (ok, next_leaf_ptr, last_var_idx)
        ok=False なら下位で割当地に失敗している。
        """

        if counters is None:
            counters = defaultdict(int)

        indent = "  " * depth   # 表示用インデント

        # ---------- ここで訪問ログ ----------
        name_tag = ""
        if isinstance(node, VariableNode):
            name_tag = f"{node.symbol}{node.index}"
        elif isinstance(node, LiteralNode):
            name_tag = '"' + "".join(node.text_tokens) + '"'

        ntype = node.__class__.__name__
        print(f"{indent}→ visit {ntype} {name_tag} (leaf_ptr={leaf_ptr})")

        # ---------- 変数ノード ----------
        if isinstance(node, VariableNode):
            # 直後にリテラルがあれば取得
            want_literal = None
            if parent_children and idx_in_parent + 1 < len(parent_children):
                nxt = parent_children[idx_in_parent + 1]
                if isinstance(nxt, LiteralNode):
                    want_literal = "".join(nxt.text_tokens)

            while leaf_ptr < len(leaves):
                pos_seq = (
                    leaves[leaf_ptr].get("xpos")
                    or leaves[leaf_ptr].get("pos")
                    or leaves[leaf_ptr].get("upos")
                    or []
                )
                flat = [p for ps in pos_seq for p in (ps if isinstance(ps, (list, tuple)) else [ps])]
                if node.pos_tag and not any(node.pos_tag in p for p in flat):
                    leaf_ptr += 1
                    continue

                # want_literal がある場合は同一葉に含むことを要求
                if want_literal and want_literal not in self._collect_text_recursive(leaves[leaf_ptr]):
                    leaf_ptr += 1
                    continue

                # --- 採用 ---
                node.leaf_idx = leaf_ptr
                counters[node.symbol] += 1
                last_var_idx = leaf_ptr
                print(f"{indent}   ↳ set {node.symbol}{counters[node.symbol]}  leaf_idx={node.leaf_idx}")

                leaf_ptr += 1
                break
            else:
                return False, leaf_ptr, last_var_idx

        # ---------- リテラルノード ----------
        elif isinstance(node, LiteralNode):
            lit = "".join(node.text_tokens)
            if last_var_idx is not None and lit in self._collect_text_recursive(leaves[last_var_idx]):
                node.leaf_idx = last_var_idx
            else:
                if lit not in self._collect_text_recursive(leaves[leaf_ptr]):
                    return False, leaf_ptr, last_var_idx
                node.leaf_idx = leaf_ptr
            print(f"{indent}   ↳ set Literal({lit}) leaf_idx={node.leaf_idx}")
            # leaf_ptr は進めない（次のノードが同葉を使う可能性）

        # ---------- その他ノード ----------
        for idx, child in enumerate(getattr(node, "children", [])):
            ok, leaf_ptr, last_var_idx = self._assign_dynamic_indices(
                child,
                leaves,
                leaf_ptr,
                counters,
                last_var_idx,
                parent_children=getattr(node, "children", []),
                idx_in_parent=idx,
                depth=depth + 1,                         # ★ 深さを +1
            )
            if not ok:
                return False, leaf_ptr, last_var_idx

        return True, leaf_ptr, last_var_idx
