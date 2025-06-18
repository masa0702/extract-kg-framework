import os  # 追加
from graphviz import Digraph
from pattern_nodes import (
    PatternNode, VariableNode, LiteralNode,
    ModifierRepeatNode, ModifierParallelNode, ParallelNode
)

def visualize_ast(
    root: PatternNode,
    out_filename: str = "ast",
    output_dir: str = ".",      # 出力先フォルダを指定可能に
    view: bool = True
):
    """
    AST を Graphviz で可視化して PNG などに出力します。
    - out_filename: 'ast' とすると ast.png が生成される
    - output_dir: 出力先フォルダ (デフォルトはカレントディレクトリ)
    - view: True で出力後に自動表示
    """
    # 出力先ディレクトリを作成（存在しなければ）
    os.makedirs(output_dir, exist_ok=True)

    dot = Digraph(name="AST", format="png")
    
    def _label(node: PatternNode) -> str:
        if isinstance(node, VariableNode):
            tag = f"-{node.pos_tag}" if node.pos_tag else ""
            return f"{node.__class__.__name__}\n[{node.symbol}{node.index}{tag}]"
        if isinstance(node, LiteralNode):
            txt = "".join(node.text_tokens)
            return f"{node.__class__.__name__}\n\"{txt}\""
        if isinstance(node, ModifierRepeatNode):
            return f"{node.__class__.__name__}\n{node.kind}{node.count}"
        if isinstance(node, ModifierParallelNode):
            return f"{node.__class__.__name__}\n{node.kind}"
        if isinstance(node, ParallelNode):
            return f"{node.__class__.__name__}"
        return node.__class__.__name__
    
    def _visit(node: PatternNode):
        nid = str(id(node))
        dot.node(nid, _label(node))
        for child in node.children:
            cid = str(id(child))
            dot.edge(nid, cid)
            _visit(child)
    
    _visit(root)
    # ファイル名とディレクトリを指定してレンダリング
    dot.render(filename=out_filename, directory=output_dir, view=view)
