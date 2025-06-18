# pattern_nodes.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Set
from collections import Counter

KIND_TO_LABEL = {"*": "連体修飾", "#": "連用修飾"}

class PatternNode:
    __slots__ = ("children",)
    def __init__(self, children: Optional[List[PatternNode]] = None):
        self.children: List[PatternNode] = children or []
        
    # --------------------------------------------------------------
    #  AST を前順（左→右）で走査する汎用イテレータ
    # --------------------------------------------------------------
    def walk(self):
        """自身を含む AST 全体を前順（pre-order）で yield する。"""
        yield self
        for c in self.children:
            yield from c.walk()

    def pretty(self, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{self.__class__.__name__}")
        for c in self.children:
            c.pretty(indent + 1)

    def debug(self, indent: int = 0) -> None:
        """ノードの型とスロット内の値を再帰的に出力"""
        prefix = "  " * indent
        cls = self.__class__
        # __slots__ の中身を表示（children は省く）
        info = {}
        for slot in getattr(cls, "__slots__", []):
            if slot == "children":
                continue
            val = getattr(self, slot, None)
            # 子ノードの場合は型のみ示す
            if isinstance(val, PatternNode):
                val = f"<{val.__class__.__name__}>"
            info[slot] = val
        print(f"{prefix}{cls.__name__} {info}")
        # 子ノードも再帰的に
        for c in self.children:
            c.debug(indent + 1)
    
    # --------------------------------------------------------------
    #  API 1: 変数と品詞制約
    # --------------------------------------------------------------
    def get_variable_constraints(self) -> List[Tuple[str, int, Optional[str]]]:
        """
        AST 内に現れる VariableNode を **左→右の出現順** に列挙し、
        (シンボル, 0-based 出現インデックス, 品詞タグ or None) のリストを返す。

        例: [("X", 0, "名詞"), ("Y", 1, "動詞")]
        """
        constraints: List[Tuple[str, int, Optional[str]]] = []
        seq_idx = 1
        for node in self.walk():
            if isinstance(node, VariableNode):
                constraints.append((node.symbol, seq_idx, node.pos_tag))
                seq_idx += 1
        return constraints

    def get_variable_info(self) -> List[Tuple[str, Optional[str], int]]:
        """Return (identifier, pos_tag, span_len) for variables in order."""
        info: List[Tuple[str, Optional[str], int]] = []

        def helper(node: PatternNode, prefix: str = "", span: int = 1):
            if isinstance(node, ModifierRepeatNode):
                helper(node.head, prefix + f"{node.kind}{node.count}", span + node.count)
                return
            if isinstance(node, VariableNode):
                ident = prefix + f"{node.symbol}{node.index}"
                info.append((ident, node.pos_tag, span))
                return
            for c in getattr(node, "children", []):
                helper(c, prefix, 1)

        helper(self)
        return info

    def get_node_span_info(self) -> List[Tuple[str, str, int, Optional[str]]]:
        """Return DFS-ordered node identifiers with span lengths.

        Each element is (ident, kind, span_len, pos_tag). ``kind`` is one of
        "modifier", "variable", "literal", or "parallel". Only modifier and
        variable nodes consume ``span_len`` bunsetsu.
        """
        result: List[Tuple[str, str, int, Optional[str]]] = []

        def dfs(node: PatternNode) -> None:
            if isinstance(node, SequenceNode):
                for c in node.elements:
                    dfs(c)
                return
            if isinstance(node, ModifierRepeatNode):
                ident = f"{node.kind}{node.count}"
                result.append((ident, "modifier", node.count, None))
                dfs(node.head)
                return
            if isinstance(node, ModifierSingleNode):
                ident = f"{node.kind}"
                result.append((ident, "modifier", 1, None))
                dfs(node.child)
                dfs(node.head)
                return
            if isinstance(node, ModifierParallelNode):
                ident = f"{node.kind}"
                result.append((ident, "modifier", 1, None))
                dfs(node.parallel_block)
                dfs(node.head)
                return
            if isinstance(node, ModifierBlockRepeatNode):
                ident = f"{node.kind}{node.count}"
                result.append((ident, "modifier", node.count, None))
                dfs(node.block)
                if node.head:
                    dfs(node.head)
                return
            if isinstance(node, VariableNode):
                ident = f"{node.symbol}{node.index}"
                result.append((ident, "variable", 1, node.pos_tag))
                return
            if isinstance(node, LiteralNode):
                ident = "".join(node.text_tokens)
                result.append((ident, "literal", 0, None))
                return
            if isinstance(node, ParallelNode):
                for idx, opt in enumerate(node.options):
                    if idx > 0:
                        result.append(("&", "parallel", 0, None))
                    dfs(opt)
                return
            for c in getattr(node, "children", []):
                dfs(c)

        dfs(self)
        return result
    
    
    # --------------------------------------------------------------
    #  API 2: 依存ラベル要求
    # --------------------------------------------------------------
    def get_dependency_label_requirements(self) -> Dict[str, int]:
        """
        AST 全体から「依存ラベル種別 : 必要本数」を集計して返す。

        * `dep_label` を持つノードをすべて対象にする
        * 修飾回数指定 (`count`) があればその数だけ加算
        * `count` を持たないノードは 1 本とみなす
        """
        counter: Counter[str] = Counter()
        for node in self.walk():
            label = getattr(node, "dep_label", None)
            if label is None:
                continue
            num = getattr(node, "count", 1)
            counter[label] += num
        return dict(counter)
    
    
    # --------------------------------------------------------------
    #  API 3: 必要依存エッジ集合
    # --------------------------------------------------------------
    def get_required_dependency_edges(self) -> Set[Tuple[int, int, str]]:
        """
        AST が要求する依存エッジ
            (from_idx, to_idx, label)  を 1 始まりインデックスで返す。
        変数は (symbol, number) で識別し、左→右出現順で採番する。
        """
        # 1. 変数 (symbol, number) → インデックス
        var2idx: Dict[Tuple[str, int], int] = {}
        idx = 1
        for node in self.walk():
            if isinstance(node, VariableNode):
                key = (node.symbol, node.index)
                if key not in var2idx:
                    var2idx[key] = idx
                    idx += 1

        # 2. Edge 収集
        edges: Set[Tuple[int, int, str]] = set()
        for node in self.walk():
            if hasattr(node, "dependency_edges"):
                for (fsym, fno), (tsym, tno), label in node.dependency_edges:
                    edges.add((var2idx[(fsym, fno)], var2idx[(tsym, tno)], label))
        return edges


    def get_literal_nodes(self, path=None):
        if path is None:
            path = []
        results = []
        # 自分がLiteralNodeの場合
        if type(self).__name__ == "LiteralNode":
            results.append((self.text_tokens, path))
        elif hasattr(self, "elements"):
            for i, child in enumerate(self.elements):
                results.extend(child.get_literal_nodes(path + [i]))
        elif hasattr(self, "parallel_block"):
            results.extend(self.parallel_block.get_literal_nodes(path + [0]))
            results.extend(self.head.get_literal_nodes(path + [1]))
        elif hasattr(self, "options"):
            for i, child in enumerate(self.options):
                results.extend(child.get_literal_nodes(path + [i]))
        elif hasattr(self, "head"):
            results.extend(self.head.get_literal_nodes(path + [0]))
        return results


class SequenceNode(PatternNode):
    __slots__ = ("elements",)
    def __init__(self, elements: List[PatternNode]):
        super().__init__(elements)
        self.elements = elements

    def __iter__(self):
        return iter(self.elements)

    def validate(self):
        pass  # 後で意味的制約チェックを追加


class VariableNode(PatternNode):
    __slots__ = ("symbol", "index", "pos_tag")
    def __init__(self, symbol: str, index: int, pos_tag: Optional[str] = None):
        super().__init__([])
        self.symbol = symbol
        self.index = index
        self.pos_tag = pos_tag

    def __str__(self):
        tag = f"-{self.pos_tag}" if self.pos_tag else ""
        return f"[{self.symbol}{self.index}{tag}]"


class LiteralNode(PatternNode):
    __slots__ = ("text_tokens",)
    def __init__(self, text_tokens: List[str]):
        super().__init__([])
        self.text_tokens = text_tokens

    def __str__(self):
        return "".join(self.text_tokens)


class ParallelNode(PatternNode):
    __slots__ = ("options",)
    def __init__(self, options: List[PatternNode]):
        super().__init__(options)
        self.options = options

    def matches_any(self, cell):
        pass  # マッチング用スタブ


class ModifierSingleNode(PatternNode):
    __slots__ = ("kind", "child", "head", "dep_label", "count")
    def __init__(self, kind, child, head, count=1, dep_label=None):
        super().__init__([child, head])
        self.kind = kind      # "*" or "#"
        self.child = child    # ← VariableNode
        self.head  = head     # ← VariableNode
        self.count = count
        self.dep_label = dep_label


class ModifierRepeatNode(PatternNode):
    __slots__ = ("kind", "count", "head", "dep_label")
    def __init__(self, kind: str, count: int, head: PatternNode):
        super().__init__([head])
        self.kind = kind    # "*" or "#"
        self.count = count  # >=1
        self.head = head
        self.dep_label = KIND_TO_LABEL[kind]


class ModifierBlockRepeatNode(PatternNode):
    """
    並列ブロック全体（または単一ノード）に
    *n / #n の修飾が “一回だけ” 付与されることを保持するノード。
    kind  : "*" または "#"
    count : 回数 (int)
    block : ParallelNode もしくは VariableNode
    head  : 修飾されるターゲット（後段の var_node）※ mod_chain 用
    """
    __slots__ = ("kind", "count", "block", "head", "dep_label")

    def __init__(self, kind, count, block, head=None):
        super().__init__([block] + ([head] if head else []))
        self.kind  = kind
        self.count = count
        self.block = block
        self.head  = head
        self.dep_label = KIND_TO_LABEL[kind]


# --------------------------------------------------------------------
#  ModifierParallelNode ― 括弧付き並列修飾
# --------------------------------------------------------------------
class ModifierParallelNode(PatternNode):
    __slots__ = ("kind", "parallel_block", "head", "dep_label", "count")

    def __init__(
        self,
        kind: str,
        parallel_block: "ParallelNode",
        head: PatternNode,
        dep_label: str | None = None,
    ):
        super().__init__([parallel_block, head])
        self.kind = kind  # "*" または "#"
        self.parallel_block = parallel_block
        self.head = head
        self.dep_label = KIND_TO_LABEL[kind]
        # 並列修飾は距離が 1 に固定
        self.count = 1



# --------------------------------------------------------------------
#  DependencyEdgeNode ― 変数どうしを結ぶ依存エッジを表現
# --------------------------------------------------------------------
class DependencyEdgeNode(PatternNode):
    """
    from_var, to_var : (symbol:str, number:int)  ← VariableNode の識別子
    dep_label        : str                       ← 依存関係ラベル
    子ノードは持たず、純粋にメタ情報のみ保持する。
    """
    __slots__ = ("from_var", "to_var", "dep_label")

    def __init__(
        self,
        from_var: Tuple[str, int],
        to_var: Tuple[str, int],
        dep_label: str,
    ):
        super().__init__([])
        self.from_var = from_var
        self.to_var = to_var
        self.dep_label = dep_label

    # --- マッチャ API が拾うためのプロパティ -------------------------
    @property
    def dependency_edges(self) -> List[Tuple[Tuple[str, int], Tuple[str, int], str]]:
        # [(from_var, to_var, label)] のリストで返す
        return [(self.from_var, self.to_var, self.dep_label)]

        

if __name__ == "__main__":
    # # --- 全ノードを使ったサンプル AST の組み立て ---
    # # 変数ノード
    # x1 = VariableNode("X", 1)
    # x2 = VariableNode("X", 2)
    # y1 = VariableNode("Y", 1, pos_tag="名詞")
    # z1 = VariableNode("Z", 1)
    # m1 = VariableNode("M", 1)
    # m2 = VariableNode("M", 2)

    # # ParallelNode: [X1|X2]
    # parallel_x = ParallelNode([x1, x2])

    # # ModifierRepeatNode: *2X1
    # mod_repeat = ModifierRepeatNode("*", 2, x1)

    # # LiteralNode: "と"
    # lit = LiteralNode(["と"])

    # # ParallelNode: [Y1|Z1]
    # parallel_y = ParallelNode([y1, z1])

    # # ModifierParallelNode: #(M1&M2)X1
    # parallel_m = ParallelNode([m1, m2])
    # mod_parallel = ModifierParallelNode("#", parallel_m, x1)

    # # 全ノードを含むシーケンス AST
    # ast = SequenceNode([
    #     mod_repeat,   # ModifierRepeatNode
    #     parallel_x,   # ParallelNode
    #     lit,          # LiteralNode
    #     parallel_y,   # ParallelNode（再利用）
    #     mod_parallel  # ModifierParallelNode
    # ])

    # # ツリー状に出力
    # ast.pretty()

# サンプルパターン：[X-名詞]を[Y-動詞]

    # ASTを手動構築
    x = VariableNode("X", 1, "名詞")
    lit = LiteralNode(["を"])
    y = VariableNode("Y", 2, "動詞")
    dep = DependencyEdgeNode("X", "Y", "項-述語")
    seq = SequenceNode([x, lit, y, dep])

    # デバッグ表示
    print("=== AST 構造 ===")
    seq.debug()

    # フィルタに使う情報を表示
    print("\n=== フィルタ情報 ===")
    print("■ 変数と品詞制約")
    for symbol, idx, pos in seq.get_variable_constraints():
        print(f"  {symbol}（{idx}番目）: 品詞={pos}")

    print("\n■ 依存ラベル要求")
    print(seq.get_dependency_label_requirements())

    print("\n■ 依存エッジ要求")
    for from_idx, to_idx, label in seq.get_required_dependency_edges():
        print(f"  {from_idx}→{to_idx} ({label})")

    print("\n■ リテラル要素")
    literal = get_literal_nodes(lit)
    print(literal)