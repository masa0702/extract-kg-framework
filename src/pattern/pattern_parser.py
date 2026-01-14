# pattern_parser.py

from lark import Lark, Transformer, v_args
from pattern_nodes import (
    PatternNode,
    SequenceNode,
    VariableNode,
    LiteralNode,
    ParallelNode,
    ModifierRepeatNode,
    ModifierParallelNode,
    ModifierSingleNode,
    ModifierBlockRepeatNode, 
)
from ast_visualizer import visualize_ast

import os

# Ensure grammar.lark can be found regardless of current working directory
GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), "grammar.lark")


class PatternParser:
    def __init__(self):
        # grammar.lark を読み込んでパーサ生成
        self._parser = Lark.open(
            GRAMMAR_FILE,
            parser="lalr",
            propagate_positions=True,
            maybe_placeholders=False,
        )
        self._transformer = PatternTransformer()

    def parse(self, text: str) -> SequenceNode:
        """
        パターン文字列を AST（SequenceNode）に変換して返す
        """
        tree = self._parser.parse(text)
        return self._transformer.transform(tree)


@v_args(inline=True)  # メソッドに渡される引数をフラット化
class PatternTransformer(Transformer):
    # start: → pattern
    def start(self, pattern):
        return pattern

    # pattern: element+
    def pattern(self, *elements):
        # SequenceNode にまとめる
        return SequenceNode(list(elements))

    # element: parallel_group | bracketed | literal
    def element(self, item):
        return item

    @v_args(inline=True)
    def par_content(self, content):
        return content
    
    # parallel_group: bracketed (AMP bracketed)+
    @v_args(inline=True)
    def parallel_group(self, *children):
        """
        grammar: parallel_group : bracketed (AMP bracketed)+
        → 同様に AMP トークンを除外して ParallelNode を構築
        """
        nodes = [c for c in children if isinstance(c, PatternNode)]
        return ParallelNode(nodes)

    # bracketed: LBRACK expr RBRACK
    def bracketed(self, _lbrack, expr, _rbrack):
        return expr

    # expr: mod_chain | parallel_expr | hash_var
    def expr(self, node):
        return node


    @v_args(inline=True)
    def modifier(self, mod_item):
        """
        grammar の
            modifier: mod_parallel | mod_repeat
        でマッチした Tree('modifier', …) をアンラップして、
        そのまま子要素（タプルかノード）を返す。
        """
        return mod_item
    
    @v_args(inline=True)
    def atom(self, item):
        """
        grammar の
            atom: var_atom | bracketed
        でマッチした Tree('atom', …) をアンラップして、
        そのまま子要素（VariableNode か AST ノード）を返す。
        """
        return item
    
    @v_args(inline=True)
    def mod_parallel_count(self, op, num, _lpar, content, _rpar):
        # (*1(A&B))Y のように “頭” が後ろに来る場合
        kind  = op.value
        cnt   = int(num.value)
        return ("BLOCK", kind, cnt, content)   # ★ タグを付けて区別

    
    # --- 2. ブロック全体を修飾する stand-alone 式 ------------------------
    # ★ “展開” せずに Block ノードを返す
    @v_args(inline=True)
    def block_repeat(self, op, num, _lpar, content, _rpar):
        kind  = op.value
        cnt   = int(num.value)
        return ModifierBlockRepeatNode(kind, cnt, content)


    # mod_chain: modifier* var_atom
    @v_args(inline=True)
    def mod_chain(self, *items):
        """
        grammar の
            mod_chain: modifier* var_atom
        で得られる items は、先頭から
            modifier, modifier, …, var_node
        という並び。ここで後ろから一つずつ適用して AST を構築する。

        - modifier() が返すものは
            • ModifierRepeatNode （hash_var 経由で既に作られたもの）
            • (kind: str, count: int) のタプル
            • (kind: str, ParallelNode) のタプル
        - 最後の items[-1] は VariableNode
        """
        *mods, var_node = items  # 最後が VariableNode
        # 右側の修飾子から順に適用
        for mod in reversed(mods):
            if isinstance(mod, ModifierRepeatNode):
                # hash_var などで作成されたノード
                mod.head = var_node
                var_node = mod
                continue

            kind, payload = mod
            if isinstance(payload, int):
                # *n, #n の回数指定
                var_node = ModifierRepeatNode(kind, payload, var_node)
            elif isinstance(payload, ParallelNode):  # *(A&B)… の並列
                # *(…)&… の括弧付き並列修飾
                var_node = ModifierParallelNode(kind, payload, var_node)
            elif isinstance(mod, tuple) and mod[0] == "BLOCK":       # ★ ここ
                _, kind, cnt, blk = mod
                var_node = ModifierBlockRepeatNode(kind, cnt, blk, head=var_node)
            else:                          # ★ ここが追加分 —— 単一要素
                # payload は VariableNode
                var_node = ModifierSingleNode(kind, payload, var_node)

        return var_node

    # mod_repeat: STAR INT | HASH INT
    def mod_repeat(self, op, num):
        kind = op.value  # "*" or "#"
        count = int(num.value)
        return (kind, count)

    # mod_parallel: (STAR|HASH) LPAR parallel_expr RPAR
    @v_args(inline=True)
    def mod_parallel(self, op, _lpar, parallel_expr, _rpar):
        """
        修飾子付き並列: e.g. *([M1]&[M2]) など
        op: Token(STAR) or Token(HASH)
        _lpar/_rpar: それぞれ "(" と ")"
        parallel_expr: ParallelNode
        """
        kind = op.value
        return (kind, parallel_expr)

    # hash_var: HASH var_atom
    def hash_var(self, _, var_node):
        # "#" のみ回数1とみなして modifier と同等扱い
        return ModifierRepeatNode("#", 1, var_node)

    # parallel_expr: atom (AMP atom)+
    @v_args(inline=True)
    def parallel_expr(self, *children):
        """
        grammar: parallel_expr : atom (AMP atom)+
        → children に AMP トークンが混ざってくるので除外する
        """
        # PatternNode のみを抽出
        atoms = [c for c in children if isinstance(c, PatternNode)]
        return ParallelNode(atoms)
    # atom: var_atom | bracketed
    # → それぞれのメソッドが返すノードをそのまま使う

    # var_atom: VAR INT pos_tag?
    def var_atom(self, var, idx, pos_tag=None):
        symbol = var.value
        index = int(idx.value)
        tag = pos_tag  # None or str
        return VariableNode(symbol, index, tag)

    # pos_tag: "-" TAG
    @v_args(inline=True)
    def pos_tag(self, tag):
        return tag.value

    # literal: /[^\[\]&*#()]+/
    def literal(self, tok):
        text = tok.value
        return LiteralNode([text])


# parser = PatternParser()
# pattern = "[*1([X1-名詞]&[X2-名詞])]を[Y1-サ変]する"
# # pattern = "[*1X1-名詞]&[*1X2-名詞]を[Y1-サ変]する"
# ast = parser.parse(pattern)
# print(ast)
# # 詳細テキスト出力
# ast.debug()
# # visualize_ast(ast, out_filename="test", view=True)

# asts = []
# patterns = [
#     "[X1-名詞]を[Y1-サ変+する]",
#     "[*1X1]を[Y1-動詞]",
#     "[*2X1]を[Y1]",
#     "[X1]を[#1Y1-動詞]",
#     "[X1]&[X2]を[Y1-サ変+する]",
#     "[X1]を[Y1-サ変]&[Y2-サ変+する]",
#     "[*1X1]&[X2]を[Y1]",
#     "[*([M1]&[M2])X1]を[Y1]",
#     "[*1*([M1]&[M2])X1]を[Y1]",
#     "[*1*([M1]&[M2])*1X1]を[Y1]",
#     "[*([M1]&[M2])*([M3]&[M4])X1]を[Y1]",
# ]

# # ④ 各パターンをパースして Success/Failure を表示
# import errno

# for p in patterns:
#     try:
#         ast = parser.parse(p)
#         asts.append(ast)
#         print(f"OK  : {p}\n")
#         visualize_ast(
#             ast,
#             out_filename=p,
#             output_dir="./test_asts",
#             view=True
#         )
#     except Exception as e:
#         # xdg-open が見つからないエラーだけ抑制
#         if isinstance(e, FileNotFoundError) \
#            and e.errno == errno.ENOENT \
#            and "xdg-open" in str(e):
#             continue

#         # その他の例外は出力
#         print(f"NG  : {p}\n    → {e}\n")


# 表示したいパターンを指定
if __name__ == "__main__":
    patterns = [
        "[X1-名詞]を[Y1-動詞]",
        "[*1X1]を[Y1]",
        "[*2X1]を[Y1]",
        "[X1]&[X2]を[Y1]する",
        "[*([M1]&[M2])X1]を[Y1]",
    ]

    parser = PatternParser()

    for p in patterns:
        try:
            print(f"\n==== パターン: {p} ====")
            ast = parser.parse(p)
            # AST構造を表示
            ast.debug()
            # フィルタで使う情報を表示
            print("■ 変数と品詞制約")
            for symbol, idx, pos in ast.get_variable_constraints():
                print(f"  {symbol}（{idx}番目）: 品詞={pos}")
            print("■ 依存ラベル要求")
            print(f"  {ast.get_dependency_label_requirements()}")
            print("■ 依存エッジ要求")
            for from_idx, to_idx, label in ast.get_required_dependency_edges():
                print(f"  {from_idx}→{to_idx} ({label})")
            print("■ リテラル要素")
            literals = ast.get_literal_nodes()
            print(literals)
        except Exception as e:
            print(f"パース失敗: {p}\n  → {e}")
