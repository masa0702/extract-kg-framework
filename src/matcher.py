from __future__ import annotations
import re
from copy import deepcopy

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
    ParallelNode,
    SequenceNode
)

# ── つなぎ語を検知する正規表現 ──────────────────────────
CONNECTIVES_REGEX = re.compile(
    r"(および|及び|ならびに|並びに|かつ|や|と|も|とも|または|あるいは|もしくは|、|，|であり|であり、|そして|され、|され|し、)"
)

# ----------------------------------------------------------------------
#  結果構造
# ----------------------------------------------------------------------
@dataclass
class MatchResult:
    i: int
    j: int
    variable_mapping: Dict[str, str]
    cell: Optional[Dict[str, Any]] = None



# ----------------------------------------------------------------------
#  CKY Matcher 本体
# ----------------------------------------------------------------------
class CKYMatcher:
    """
    span>1 の CKY セル（cell["candidates"] 内の各候補）に対して
    パターン AST を照合し，合致するものを返す。
    """

    def __init__(self, pattern_ast: PatternNode, *, verbose: bool = False) -> None:
        self.pattern_ast = pattern_ast
        self.verbose = verbose     # ← 追加
        self._all_vars = list(self._collect_variable_nodes(self.pattern_ast))
        self._assign_seq_ids()

    # -------------------------------------------------------------
    #  ログ出力（verbose=True のときだけ表示）
    # -------------------------------------------------------------
    def _log(self, *msg):
        if self.verbose:
            print("[CKYMatcher]", *msg)

    # ------------------------------------------------------------------
    #  CKY 表全体を走査してパターンに合う variable_mapping を返す
    #  - list[list[...]] フォーマットにも
    #  - クラスに .table があるフォーマットにも対応
    # ------------------------------------------------------------------
    def match_table(self, cky) -> List[Dict[str, str]]:
        """
        Parameters
        ----------
        cky : object
            - `cky.table` が 2D list の CKYTable クラス
            - あるいは 2D list そのもの
            いずれにも対応する。
        Returns
        -------
        List[Dict[str, str]]
            パターンにマッチした variable_mapping のリスト
        """
        # ---------- 1) 内部 2D 配列を取り出す ----------
        if hasattr(cky, "table"):
            mat = cky.table
        else:                         # すでに list[list]
            mat = cky

        # mat[0] は列ヘッダ、mat[i][0] は行ヘッダという前提
        n = len(mat) - 1              # 文節数

        results: List[Dict[str, str]] = []

        # ---------- 2) span 長が長い方から走査 ----------
        for span in range(n, 0, -1):          # n, n-1, …, 1
            for i in range(1, n - span + 2):  # 開始インデックス (1-based)
                j = i + span - 1              # 終了インデックス
                cell = mat[i][j]

                # 空セルや整数 0 ヘッダなどはスキップ
                if not isinstance(cell, dict) or "candidates" not in cell:
                    continue
                if not cell["candidates"]:
                    continue

                # ---------- 3) 中の候補を評価 ----------
                # matcher.py  match_table 内 「for cand in cell['candidates']」 直後を修正
                for cand in cell["candidates"]:
                    mapping = self._match_candidate(cand)
                    if mapping is not None:
                        results.append(
                            MatchResult(i=i, j=j, variable_mapping=mapping, cell=cand)
                        )



        return results



    # ---------- internal ----------
    # -------------------------------------------------------------
    #  候補セル 1 個を評価
    # -------------------------------------------------------------
    def _match_candidate(self, cand: Dict[str, Any]) -> Optional[Dict[str, str]]:
        # ① 直前の評価で付いた leaf_idx をリセット
        self._clear_leaf_indices(self.pattern_ast)

        # ② 既存処理
        leaves = self._collect_leaves(cand)

        ok, _, _ = self._assign_dynamic_indices(self.pattern_ast, leaves)
        if not ok:
            self._log("A: dynamic-index failed")
            return None

        if not self._dependency_label_filter(cand):
            self._log("B: dependency-label filter failed")
            return None
        if not self._literal_filter(cand, leaves):
            self._log("C: literal filter failed")
            return None

        res = self._pos_and_variable_filter(cand, leaves)
        if res is None:
            self._log("D: pos/variable filter failed")
        else:
            self._log("✓ matched", res)
        return res


    # --- phase-1 : 依存ラベル ---
    def _dependency_label_filter(self, cand: Dict[str, Any]) -> bool:
        required = self.pattern_ast.get_dependency_label_requirements() or {}
        if not required:
            return True
        actual = Counter(self._collect_dep_labels(cand))
        return all(actual.get(lbl, 0) >= need for lbl, need in required.items())

    # --- phase-2 : リテラル ---
    def _literal_filter(self, cand: Dict[str, Any], leaves) -> bool:
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

            concat = "".join(self._collect_text_recursive(leaves[idx])
                             for idx in idx_list)
            if lit not in concat:
                print(f"    ✗ literal '{lit}' not in span[{idx_list}]='{concat}'")
                return False
        return True

    # --- phase-3 : 品詞 + 変数 ---
    def _pos_and_variable_filter(self, cand: Dict[str, Any], leaves) -> Optional[Dict[str, str]]:
        var_constraints = self.pattern_ast.get_variable_constraints()
        varmap: Dict[str, str] = {}
        counter = defaultdict(int)

        for sym, leaf_idx, pos_tag in var_constraints:
            if leaf_idx >= len(leaves):
                return None
            leaf = leaves[leaf_idx]
            surface = leaf.get("candidate") or leaf.get("text", "")

            if pos_tag:
                pos_seq = (leaf.get("xpos") or leaf.get("pos") or
                           leaf.get("upos") or [])
                flat = [p for seq in pos_seq
                        for p in (seq if isinstance(seq, (list, tuple)) else [seq])]
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
    def _assign_dynamic_indices(
            self,
            node: PatternNode,
            leaves: List[Dict[str, Any]],
            leaf_ptr: int = 0,
            counters: Optional[defaultdict] = None,
            last_var_idx: Optional[int] = None,
            parent_children: Optional[list] = None,
            idx_in_parent: int = -1,
            depth: int = 0,
            parent_node: Optional[PatternNode] = None,   # ← ここを None にする
    ) -> tuple[bool, int, Optional[int]]:


        if counters is None:
            counters = defaultdict(int)

        # matcher.py  内 _assign_dynamic_indices() より
        # -------------- ParallelNode --------------
        elif isinstance(node, ParallelNode):
            import itertools

            # ---------- permutation を総当たり ----------
            for perm in itertools.permutations(node.options):
                snap = self._snapshot(leaf_ptr, counters)
                lp_tmp, last_tmp = leaf_ptr, last_var_idx
                success = True

                for opt in perm:
                    idx_child = perm.index(opt)
                    ok, lp_tmp, last_tmp = self._assign_dynamic_indices(
                        opt, leaves, lp_tmp, counters, last_tmp, parent_children=perm, idx_in_parent=idx_child,
                        depth=depth + 1, parent_node=node
                    )
                    if not ok:
                        success = False
                        break

                if success:
                    leaf_ptr, last_var_idx = lp_tmp, last_tmp
                    break  # ← マッチ成功
                else:
                    leaf_ptr = self._restore(snap, counters)
            else:
                return False, leaf_ptr, last_var_idx

        
        # -------------- Modifier*Node --------------
        elif isinstance(node, (ModifierSingleNode,
                            ModifierRepeatNode,
                            ModifierBlockRepeatNode)):
            # min_times, max_times のチェックは count==1 だけ実装
            # *1 なので必ず 1 回 child をマッチさせる
            child = node.child if hasattr(node, "child") else None
            if child is None:
                # RepeatBlock の場合は children list
                children = node.children
            else:
                children = [child]

            saved = self._snapshot(leaf_ptr, counters)
            for ch in children:
                ok, leaf_ptr, last_var_idx = self._assign_dynamic_indices(
                    ch, leaves, leaf_ptr, counters,
                    last_var_idx,                 # ← ここを修正
                    parent_children=children, idx_in_parent=0,
                    depth=depth + 1, parent_node=node
                )
                if not ok:
                    leaf_ptr = self._restore(saved, counters)
                    return False, leaf_ptr, last_var_idx
            # === 追加ここから ===
            # 子ども側から帰ってきた last_var_idx が int でなければ
            # 親側の last_var_idx を保持（上書きしない）
            if not isinstance(last_var_idx, int):
                last_var_idx = saved[1]   # snap に保存していた leaf_ptr が int
            # === 追加ここまで ===
            return True, leaf_ptr, last_var_idx
        
        
        # ---------- ModifierRepeatNode (‘*’) ----------
        if isinstance(node, ModifierRepeatNode) and node.kind == "*":
            orig_ptr = leaf_ptr
            for rep in range(0, min(node.count, 5) + 1):
                cur_ptr = orig_ptr
                ok = True
                for _ in range(rep):
                    ok, cur_ptr, last_var_idx = self._assign_dynamic_indices(
                        node.head, leaves, cur_ptr, counters, last_var_idx,
                        parent_children=[node.head], idx_in_parent=0,
                        depth=depth + 1,parent_node=node)
                    if not ok:
                        break
                if ok:
                    leaf_ptr = cur_ptr
                    # --- 追加 ---
                    if not isinstance(last_var_idx, int):
                        last_var_idx = orig_ptr
                    # ------------

                    return True, leaf_ptr, last_var_idx
            return False, leaf_ptr, last_var_idx

    # -------------------------------------------------------------
    #  VariableNode で leaf を割り当てる部分
    # -------------------------------------------------------------
        # ---------- VariableNode ----------
        if isinstance(node, VariableNode):
            # すでに leaf が確定していれば再利用して次へ
            if node.leaf_idx is not None:
                self._log(f"reuse {node.symbol}{node.index} -> leaf[{node.leaf_idx}]")
                return True, leaf_ptr, last_var_idx

            while leaf_ptr < len(leaves):
                cur_text = self._collect_text_recursive(leaves[leaf_ptr])

                # Parallel 左辺なら接続詞を要求
                in_parallel = isinstance(parent_node, ParallelNode)
                if in_parallel and parent_children and idx_in_parent < len(parent_children) - 1:
                    if CONNECTIVES_REGEX.search(cur_text) is None:
                        self._log(f"skip leaf[{leaf_ptr}]='{cur_text}' (no connective)")
                        leaf_ptr += 1
                        continue

                # pos_tag 条件
                if node.pos_tag:
                    pos_seq = (leaves[leaf_ptr].get("xpos")
                               or leaves[leaf_ptr].get("pos")
                               or leaves[leaf_ptr].get("upos") or [])
                    flat = [p for seq in pos_seq
                            for p in (seq if isinstance(seq, (list, tuple)) else [seq])]
                    if not any(node.pos_tag in p for p in flat):
                        self._log(f"skip leaf[{leaf_ptr}]='{cur_text}' (pos_tag mismatch)")
                        leaf_ptr += 1
                        continue

                # 直後 Literal 要件
                want_literal = None
                if isinstance(parent_node, SequenceNode):
                    if idx_in_parent + 1 < len(parent_children):
                        nxt = parent_children[idx_in_parent + 1]
                        if isinstance(nxt, LiteralNode):
                            want_literal = "".join(nxt.text_tokens)
                if want_literal and want_literal not in cur_text:
                    self._log(f"skip leaf[{leaf_ptr}]='{cur_text}' (want_literal '{want_literal}')")
                    leaf_ptr += 1
                    continue

                # --- 採用 ---
                node.leaf_idx = leaf_ptr
                self._log(f"assign {node.symbol}{node.index} -> leaf[{leaf_ptr}]='{cur_text}'")
                last_var_idx = leaf_ptr
                leaf_ptr += 1
                break
            else:
                self._log(f"fail: cannot assign {node.symbol}{node.index}")
                return False, leaf_ptr, last_var_idx


        # ---------- LiteralNode ----------
        elif isinstance(node, LiteralNode):
            lit = "".join(node.text_tokens)

            # --- ① 直前の変数 leaf を優先 ---
            if last_var_idx is not None:
                if 0 <= last_var_idx < len(leaves):         # ★ 安全チェックを追加 ★
                    if lit in self._collect_text_recursive(leaves[last_var_idx]):
                        node.leaf_idx = last_var_idx
                        return True, leaf_ptr, last_var_idx
                # インデックスが不正なら無視して次へ

            # --- ② 現在の leaf_ptr を確認 ---
            if leaf_ptr < len(leaves) and lit in self._collect_text_recursive(leaves[leaf_ptr]):
                node.leaf_idx = leaf_ptr
                return True, leaf_ptr, last_var_idx

            return False, leaf_ptr, last_var_idx


        # ---------- その他ノード ----------
        for idx, child in enumerate(getattr(node, "children", [])):
            ok, leaf_ptr, last_var_idx = self._assign_dynamic_indices(
                child, leaves, leaf_ptr, counters, last_var_idx,
                parent_children=getattr(node, "children", []),
                idx_in_parent=idx, depth=depth + 1,parent_node=node)
            if not ok:
                return False, leaf_ptr, last_var_idx

        return True, leaf_ptr, last_var_idx

    # -------------------------------------------------------------
    #  leaf_idx を再帰的にクリア
    # -------------------------------------------------------------
    def _clear_leaf_indices(self, node: PatternNode):
        if isinstance(node, VariableNode):
            node.leaf_idx = None
        for child in getattr(node, "children", []):
            self._clear_leaf_indices(child)

    # ---------------------------------------------------------
    #  変数ノード一覧を事前に集めておく  (__init__ の末尾で呼び出し)
    # ---------------------------------------------------------
    def _collect_variable_nodes(self, node):
        from pattern_nodes import VariableNode
        if isinstance(node, VariableNode):
            yield node
        for ch in getattr(node, "children", []):
            yield from self._collect_variable_nodes(ch)

    # ---------------------------------------------------------
    #  スナップショットを取得
    # ---------------------------------------------------------
    def _snapshot(self, leaf_ptr, counters):
        leaf_indices = [n.leaf_idx for n in self._all_vars]
        return (leaf_indices, leaf_ptr, deepcopy(counters))

    # ---------------------------------------------------------
    #  スナップショットから復元
    # ---------------------------------------------------------
    def _restore(self, snap, counters):
        leaf_indices, lp_saved, ctr_saved = snap
        for n, idx in zip(self._all_vars, leaf_indices):
            n.leaf_idx = idx
        counters.clear()
        counters.update(ctr_saved)
        return lp_saved          # ← 呼び出し側で leaf_ptr に代入
