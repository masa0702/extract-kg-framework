import pandas as pd
import zss
import os
from datetime import datetime
from pattern_parser import PatternParser
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

parser = PatternParser()

class ZssNode(zss.Node):
    def __init__(self, label, children=None):
        super().__init__(label)
        if children:
            for c in children:
                self.addkid(c)

def ast_to_zss(node):
    # ラベル作成
    if isinstance(node, VariableNode):
        label = f"VariableNode:{node.symbol},{node.pos_tag or ''}"
        children = []
    elif isinstance(node, LiteralNode):
        label = f"LiteralNode:{''.join(node.text_tokens)}"
        children = []
    elif isinstance(node, SequenceNode):
        label = "SequenceNode"
        children = node.elements
    elif isinstance(node, ParallelNode):
        label = "ParallelNode"
        children = node.options
    elif isinstance(node, ModifierParallelNode):
        label = f"ModifierParallelNode:{node.kind}:{node.dep_label}"
        children = [node.parallel_block, node.head]
    elif isinstance(node, ModifierRepeatNode):
        label = f"ModifierRepeatNode:{node.kind}:{node.count}:{node.dep_label}"
        children = [node.head]
    elif isinstance(node, ModifierBlockRepeatNode):
        label = f"ModifierBlockRepeatNode:{node.kind}:{node.count}:{node.dep_label}"
        children = [node.block] + ([node.head] if node.head else [])
    else:
        # 汎用
        label = node.__class__.__name__
        children = getattr(node, "children", [])
    zss_children = [ast_to_zss(child) for child in children]
    return ZssNode(label, zss_children)



def tree_size(node):
    return 1 + sum(tree_size(child) for child in node.children)

def tree_labels(node):
    labels = set([node.label])
    for child in node.children:
        labels.update(tree_labels(child))
    return labels

def f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0

def eval_csv(csv_path):
    df = pd.read_csv(csv_path)
    total = len(df)
    parseable = 0
    ted_sum = 0
    ted_count = 0
    f1_sum = 0
    f1_count = 0

    # 各行ごとの詳細結果も記録
    line_results = []

    for idx, row in df.iterrows():
        gold = row['gold_pattern']
        pred = row['pattern']

        gold_tree, pred_tree = None, None
        gold_ok, pred_ok = 0, 0
        gold_err, pred_err = '', ''
        ted_norm = None
        f1 = None

        # goldパース
        try:
            gold_tree = parser.parse(gold)
            gold_ok = 1
        except Exception as e:
            gold_ok = 0
            gold_err = str(e)

        # predパース
        try:
            pred_tree = parser.parse(pred)
            pred_ok = 1
        except Exception as e:
            pred_ok = 0
            pred_err = str(e)

        # 両方成功時のみTEDやF1を計算
        # 例：パース成功時のみ
        if gold_ok and pred_ok:
            parse_flag = 1
            parseable += 1
            try:
                # AST→zssノード変換
                gold_zss = ast_to_zss(gold_tree)
                pred_zss = ast_to_zss(pred_tree)
                # TED
                ted = zss.simple_distance(gold_zss, pred_zss)
                max_size = max(tree_size(gold_zss), tree_size(pred_zss))
                ted_norm = 1 - ted / max_size if max_size else 1
                ted_sum += ted_norm
                ted_count += 1

                # ラベルF1
                gold_labels = tree_labels(gold_zss)
                pred_labels = tree_labels(pred_zss)
                tp = len(gold_labels & pred_labels)
                fp = len(pred_labels - gold_labels)
                fn = len(gold_labels - pred_labels)
                f1 = f1_score(tp, fp, fn)
                f1_sum += f1
                f1_count += 1
            except Exception as e:
                ted_norm = None
                f1 = None
        else:
            parse_flag = 0


        # 詳細結果の記録
        line_results.append({
            'idx': idx,
            'gold_pattern': gold,
            'transform_pattern': pred,
            'parseable': parse_flag,
            'gold_parse': gold_ok,
            'pred_parse': pred_ok,
            'gold_err': gold_err,
            'pred_err': pred_err,
            'ted_norm': ted_norm,
            'node_label_f1': f1
        })

    parseability = parseable / total if total else 0
    avg_ted      = ted_sum / ted_count if ted_count else 0
    avg_f1       = f1_sum / f1_count if f1_count else 0

    summary = (
        f'パース成功率（parseability）：{parseability:.2%}\n'
        f'平均構文木編集距離スコア（TED正規化平均）：{avg_ted:.3f}\n'
        f'平均ノードラベルF1（Tree Label F1）：{avg_f1:.3f}\n'
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = "./results_eval_text2pattern"
    os.makedirs(save_dir, exist_ok=True)
    out_filename = os.path.join(save_dir, f'eval_result_{timestamp}.txt')

    with open(out_filename, 'w', encoding='utf-8') as f:
        f.write(f'# 評価日時: {datetime.now()}\n')
        f.write(f'# 入力ファイル: {csv_path}\n')
        f.write('\n=== 集計結果（summary） ===\n')
        f.write(summary)
        f.write('\n=== 各行ごとの詳細 ===\n')
        f.write('idx\tparseable\tgold_parse\tpred_parse\tted_norm\tnode_label_f1\tgold_pattern\ttransform_pattern\tgold_err\tpred_err\n')
        for r in line_results:
            f.write(f"{r['idx']}\t{r['parseable']}\t{r['gold_parse']}\t{r['pred_parse']}\t"
                    f"{'' if r['ted_norm'] is None else round(r['ted_norm'],3)}\t"
                    f"{'' if r['node_label_f1'] is None else round(r['node_label_f1'],3)}\t"
                    f"{r['gold_pattern']}\t{r['transform_pattern']}\t"
                    f"{r['gold_err']}\t{r['pred_err']}\n")
    print(f'評価結果を {out_filename} に保存しました。')

if __name__ == '__main__':
    csv_path = "../gemini_api/results/result_text_to_pattern_eval_20_gpt-4o-mini_ver4.5.csv"
    eval_csv(csv_path)
