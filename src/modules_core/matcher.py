from __future__ import annotations

import itertools
import re
from copy import deepcopy
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pattern.pattern_nodes import (
    PatternNode,
    VariableNode,
    ModifierSingleNode,
    ModifierParallelNode,
    ModifierRepeatNode,
    ModifierBlockRepeatNode,
    LiteralNode,
    ParallelNode,
    SequenceNode,
    GapNode,
)

# 接続詞を検知する正規表現
CONNECTIVES_REGEX = re.compile(
    r"(および|及び|ならびに|並びに|かつ|や|と|も|とも|または|あるいは|もしくは|、|，|であり|であり、|そして|され、|され|し、)"
)


@dataclass
class MatchResult:
    i: int
    j: int
    variable_mapping: Dict[str, str]
    cell: Optional[Dict[str, Any]] = None


class CKYMatcher:
    """CKYセルの候補に対してパターンASTを照合する。"""

    def __init__(self, pattern_ast: PatternNode, *, verbose: bool = False) -> None:
        self.pattern_ast = pattern_ast
        self.verbose = verbose
        self._all_leaf_nodes = list(self._collect_leaf_nodes(self.pattern_ast))
        self._assign_seq_ids()

    # -------------------------------------------------------------
    #  ログ出力（verbose=True のときだけ表示）
    # -------------------------------------------------------------
    def _log(self, *msg):
        if self.verbose:
            print("[CKYMatcher]", *msg)

    def match_table(self, cky) -> List[Dict[str, str]]:
        """Return MatchResult list for cells containing pattern matches."""
        if hasattr(cky, "table"):
            mat = cky.table
        else:
            mat = cky

        n = len(mat) - 1

        results: List[Dict[str, str]] = []

        # span 長が長い方から走査
        for span in range(n, 0, -1):
            for i in range(1, n - span + 2):
                j = i + span - 1
                cell = mat[i][j]

                if not isinstance(cell, dict) or "candidates" not in cell:
                    continue
                if not cell["candidates"]:
                    continue

                for cand in cell["candidates"]:
                    mappings = self._match_candidate_all(cand)
                    for mapping in mappings:
                        results.append(
                            MatchResult(i=i, j=j, variable_mapping=mapping, cell=cand)
                        )

        return results



    # ---------- internal ----------
    # -------------------------------------------------------------
    #  候補セル 1 個を評価
    # -------------------------------------------------------------
    def _match_candidate_all(self, cand: Dict[str, Any]) -> List[Dict[str, str]]:
        """Evaluate one candidate subtree and return all var mappings."""
        self._clear_leaf_indices(self.pattern_ast)
        leaves = self._collect_leaves(cand)
        matches: List[Dict[str, str]] = []
        seen = set()

        for _lp, _last in self._iter_assignments(self.pattern_ast, leaves):
            if not self._dependency_label_filter(cand):
                self._log("B: dependency-label filter failed")
                continue
            if not self._literal_filter(cand, leaves):
                self._log("C: literal filter failed")
                continue

            res = self._pos_and_variable_filter(leaves)
            if res is None:
                self._log("D: pos/variable filter failed")
                continue
            key = frozenset(res.items())
            if key in seen:
                continue
            seen.add(key)
            self._log("✓ matched", res)
            matches.append(res)

        if not matches:
            self._log("A: dynamic-index failed")
        return matches


    # --- phase-1 : 依存ラベル ---
    def _dependency_label_filter(self, cand: Dict[str, Any]) -> bool:
        required = self.pattern_ast.get_dependency_label_requirements() or {}
        if not required:
            return True
        actual = Counter(self._collect_dep_labels(cand))
        return all(actual.get(lbl, 0) >= need for lbl, need in required.items())

    # --- phase-2 : リテラル ---
    def _literal_filter(self, cand: Dict[str, Any], leaves: List[Dict[str, Any]]) -> bool:
        literal_nodes = self.pattern_ast.get_literal_nodes()
        if not literal_nodes:
            return True

        total = len(leaves)

        for tokens, leaf_idx in literal_nodes:
            lit = "".join(tokens)
            if leaf_idx is None:
                if lit not in cand.get("text", ""):
                    return False
                continue

            idx_list = [leaf_idx] if isinstance(leaf_idx, int) else leaf_idx
            if any(idx < 0 or idx >= total for idx in idx_list):
                return False

            concat = "".join(self._collect_text_recursive(leaves[idx]) for idx in idx_list)
            if lit not in concat:
                return False
        return True

    # --- phase-3 : 品詞 + 変数 ---
    def _pos_and_variable_filter(self, leaves: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        var_constraints = self.pattern_ast.get_variable_constraints()
        varmap: Dict[str, str] = {}
        counter = defaultdict(int)

        for sym, leaf_idx, pos_tag in var_constraints:
            if leaf_idx >= len(leaves):
                return None
            leaf = leaves[leaf_idx]
            surface = leaf.get("candidate") or leaf.get("text", "")

            if pos_tag:
                flat = self._flatten_pos_seq(leaf.get("xpos") or leaf.get("pos") or leaf.get("upos") or [])
                if not any(pos_tag in p for p in flat):
                    return None

            counter[sym] += 1
            varmap[f"{sym}{counter[sym]}"] = surface

        # --- 接続詞・補助動詞の末尾除去 ---
        def _strip_surface(s: str) -> str:
            for suf in (
                "であり、", "であり，", "であり",
                "である、", "である，", "である",
                "でもありました", "でもあります",
                "でもあり、", "でもあり", "でもある",
                "でも",
                "でした", "です", "でありました", "であります",
                "および", "及び", "ならびに", "並びに", "かつ",
                "や", "と", "とも", "と共に", "または", "あるいは", "もしくは",
                "、", "，", "でもあります。", "によって"
            ):
                if s.endswith(suf):
                    return s[:-len(suf)]
            return s

        return {k: _strip_surface(v) for k, v in varmap.items()}

    # ---------- utility ----------
    def _collect_dep_labels(self, node: Dict[str, Any]) -> List[str]:
        labels = []
        if isinstance(node, dict) and "dependency" in node:
            lbl = node["dependency"].get("label")
            if lbl:
                labels.append(lbl)
        for side in ("left", "right"):
            if isinstance(node, dict) and side in node:
                labels.extend(self._collect_dep_labels(node[side]))
        return labels

    def _collect_leaves(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(node, dict) and "left" in node and "right" in node:
            return (self._collect_leaves(node["left"]) +
                    self._collect_leaves(node["right"]))
        return [node]

    def _collect_text_recursive(self, node: Dict[str, Any]) -> str:
        if node is None or not isinstance(node, dict):
            return ""
        if "left" in node and "right" in node:
            return (self._collect_text_recursive(node["left"]) +
                    self._collect_text_recursive(node["right"]))
        return node.get("text", "")

    def _flatten_pos_seq(self, pos_seq: Iterable[Any]) -> List[str]:
        flat: List[str] = []
        for seq in pos_seq:
            if isinstance(seq, str):
                flat.append(seq)
                continue
            if isinstance(seq, (list, tuple)):
                flat.extend(seq)
            else:
                flat.append(seq)
        return flat

    # ------------------------------------------------------------------
    #  DFS 前順で seq_id を付与
    # ------------------------------------------------------------------
    def _assign_seq_ids(self) -> None:
        cur = 1
        for node in self.pattern_ast.walk():
            if isinstance(node, (VariableNode,
                                 ModifierSingleNode,
                                 ModifierParallelNode)):
                node.seq_id = cur
                cur += 1
            elif isinstance(node, ModifierRepeatNode):
                node.seq_id = cur
                cur += node.count
            elif isinstance(node, ModifierBlockRepeatNode):
                node.seq_id = None
            elif isinstance(node, LiteralNode):
                node.seq_id = None
            else:
                node.seq_id = None

    # ------------------------------------------------------------------
    #  葉インデックスを付与
    # ------------------------------------------------------------------
    def _iter_assignments(
        self,
        node: PatternNode,
        leaves: List[Dict[str, Any]],
        leaf_ptr: int = 0,
        counters: Optional[defaultdict] = None,
        last_var_idx: Optional[int] = None,
        parent_children: Optional[Sequence[PatternNode]] = None,
        idx_in_parent: int = -1,
        depth: int = 0,
        parent_node: Optional[PatternNode] = None,
        force_leaf_ptr: bool = False,
    ):
        if counters is None:
            counters = defaultdict(int)

        # ParallelNode: options の並び順で評価（並列ブロック内は連続要素のみ）
        if isinstance(node, ParallelNode):
            options = list(node.options)
            snap = self._snapshot(leaf_ptr, counters)
            force_flags = [False] + [True] * (len(options) - 1)
            for state in self._iter_children(
                options,
                leaves,
                leaf_ptr,
                counters,
                last_var_idx,
                parent_node=node,
                force_flags=force_flags,
            ):
                yield state
            self._restore(snap, counters)
            return

        # Modifier*Node
        elif isinstance(node, (ModifierSingleNode, ModifierRepeatNode, ModifierBlockRepeatNode)):
            children: Sequence[PatternNode]
            child = getattr(node, "child", None)
            children = node.children if child is None else [child]

            saved = self._snapshot(leaf_ptr, counters)
            for state in self._iter_children(
                children,
                leaves,
                leaf_ptr,
                counters,
                last_var_idx,
                parent_node=node,
            ):
                yield state
            self._restore(saved, counters)
            return

        # ModifierRepeatNode (‘*’) の可変長展開
        if isinstance(node, ModifierRepeatNode) and node.kind == "*":
            orig_ptr = leaf_ptr
            for rep in range(0, min(node.count, 5) + 1):
                snap = self._snapshot(leaf_ptr, counters)
                for state in self._iter_repeat(
                    node.head,
                    rep,
                    leaves,
                    orig_ptr,
                    counters,
                    last_var_idx,
                    parent_node=node,
                ):
                    yield state
                self._restore(snap, counters)
            return

        # VariableNode
        if isinstance(node, VariableNode):
            if node.leaf_idx is not None:
                self._log(f"reuse {node.symbol}{node.index} -> leaf[{node.leaf_idx}]")
                yield leaf_ptr, last_var_idx
                return

            ptr = leaf_ptr
            # 直前が LiteralNode なら、次の leaf に限定して割当てる
            if parent_children and idx_in_parent > 0:
                prev = parent_children[idx_in_parent - 1]
                if isinstance(prev, LiteralNode):
                    force_leaf_ptr = True

            while ptr < len(leaves):
                cur_text = self._collect_text_recursive(leaves[ptr])

                in_parallel = isinstance(parent_node, ParallelNode)
                if in_parallel and parent_children and idx_in_parent < len(parent_children) - 1:
                    if CONNECTIVES_REGEX.search(cur_text) is None:
                        self._log(f"skip leaf[{ptr}]='{cur_text}' (no connective)")
                        ptr += 1
                        continue

                if node.pos_tag:
                    flat = self._flatten_pos_seq(
                        leaves[ptr].get("xpos")
                        or leaves[ptr].get("pos")
                        or leaves[ptr].get("upos")
                        or []
                    )
                    if not any(node.pos_tag in p for p in flat):
                        self._log(f"skip leaf[{ptr}]='{cur_text}' (pos_tag mismatch)")
                        if force_leaf_ptr:
                            break
                        ptr += 1
                        continue

                want_literal = None
                if isinstance(parent_node, SequenceNode) and parent_children:
                    if idx_in_parent + 1 < len(parent_children):
                        nxt = parent_children[idx_in_parent + 1]
                        if isinstance(nxt, LiteralNode):
                            want_literal = "".join(nxt.text_tokens)
                if want_literal and want_literal not in cur_text:
                    self._log(f"skip leaf[{ptr}]='{cur_text}' (want_literal '{want_literal}')")
                    if force_leaf_ptr:
                        break
                    ptr += 1
                    continue

                node.leaf_idx = ptr
                self._log(f"assign {node.symbol}{node.index} -> leaf[{ptr}]='{cur_text}'")
                yield ptr + 1, ptr
                node.leaf_idx = None
                if force_leaf_ptr:
                    break
                ptr += 1
            return

        # LiteralNode
        elif isinstance(node, LiteralNode):
            lit = "".join(node.text_tokens)
            if last_var_idx is not None and 0 <= last_var_idx < len(leaves):
                if lit in self._collect_text_recursive(leaves[last_var_idx]):
                    node.leaf_idx = last_var_idx
                    yield leaf_ptr, last_var_idx
                    node.leaf_idx = None

            if leaf_ptr < len(leaves) and lit in self._collect_text_recursive(leaves[leaf_ptr]):
                node.leaf_idx = leaf_ptr
                yield leaf_ptr, last_var_idx
                node.leaf_idx = None
            return

        # GapNode: 文節スキップ幅を強制
        elif isinstance(node, GapNode):
            min_skip = max(0, int(getattr(node, "min_skip", 0)))
            max_skip = max(min_skip, int(getattr(node, "max_skip", min_skip)))
            for skip in range(min_skip, max_skip + 1):
                next_ptr = leaf_ptr + skip
                if next_ptr <= len(leaves):
                    yield next_ptr, last_var_idx
            return

        # その他ノード
        children = getattr(node, "children", [])
        for state in self._iter_children(
            children,
            leaves,
            leaf_ptr,
            counters,
            last_var_idx,
            parent_node=node,
        ):
            yield state
        return

    # -------------------------------------------------------------
    #  leaf_idx を再帰的にクリア
    # -------------------------------------------------------------
    def _clear_leaf_indices(self, node: PatternNode):
        if isinstance(node, (VariableNode, LiteralNode)):
            node.leaf_idx = None
        for child in getattr(node, "children", []):
            self._clear_leaf_indices(child)

    # ---------------------------------------------------------
    #  変数ノード一覧を事前に集めておく  (__init__ の末尾で呼び出し)
    # ---------------------------------------------------------
    def _collect_leaf_nodes(self, node: PatternNode):
        if isinstance(node, (VariableNode, LiteralNode)):
            yield node
        for ch in getattr(node, "children", []):
            yield from self._collect_leaf_nodes(ch)

    # ---------------------------------------------------------
    #  スナップショットを取得
    # ---------------------------------------------------------
    def _snapshot(self, leaf_ptr, counters):
        leaf_indices = [n.leaf_idx for n in self._all_leaf_nodes]
        return (leaf_indices, leaf_ptr, deepcopy(counters))

    # ---------------------------------------------------------
    #  スナップショットから復元
    # ---------------------------------------------------------
    def _restore(self, snap, counters):
        leaf_indices, lp_saved, ctr_saved = snap
        for n, idx in zip(self._all_leaf_nodes, leaf_indices):
            n.leaf_idx = idx
        counters.clear()
        counters.update(ctr_saved)
        return lp_saved          # ← 呼び出し側で leaf_ptr に代入

    def _iter_children(
        self,
        children: Sequence[PatternNode],
        leaves: List[Dict[str, Any]],
        leaf_ptr: int,
        counters: defaultdict,
        last_var_idx: Optional[int],
        parent_node: Optional[PatternNode],
        force_flags: Optional[Sequence[bool]] = None,
    ):
        def _walk(idx: int, lp: int, last: Optional[int]):
            if idx >= len(children):
                yield lp, last
                return
            child = children[idx]
            force_leaf_ptr = False
            if force_flags and idx < len(force_flags):
                force_leaf_ptr = force_flags[idx]
            for lp2, last2 in self._iter_assignments(
                child,
                leaves,
                lp,
                counters,
                last,
                parent_children=children,
                idx_in_parent=idx,
                depth=0,
                parent_node=parent_node,
                force_leaf_ptr=force_leaf_ptr,
            ):
                yield from _walk(idx + 1, lp2, last2)

        yield from _walk(0, leaf_ptr, last_var_idx)

    def _iter_repeat(
        self,
        head: PatternNode,
        repeat: int,
        leaves: List[Dict[str, Any]],
        leaf_ptr: int,
        counters: defaultdict,
        last_var_idx: Optional[int],
        parent_node: Optional[PatternNode],
    ):
        def _walk(rep_idx: int, lp: int, last: Optional[int]):
            if rep_idx >= repeat:
                yield lp, last
                return
            for lp2, last2 in self._iter_assignments(
                head,
                leaves,
                lp,
                counters,
                last,
                parent_children=[head],
                idx_in_parent=0,
                depth=0,
                parent_node=parent_node,
            ):
                yield from _walk(rep_idx + 1, lp2, last2)

        yield from _walk(0, leaf_ptr, last_var_idx)
