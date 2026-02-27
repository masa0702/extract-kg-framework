"""
任意の文とパターンを入力して CKYMatcher を実行する統合テスト用スクリプト。

例:
    python tests/matcher_integration_runner.py \\
        --sentence "太郎はリンゴを買う" \\
        --pattern "[X1]は[Y2]を買う"

SpaCy/GiNZA で形態素解析し、文節リストから CKY 表を作成したうえで、
最長スパンに単純な右枝分かれ候補木を挿入してマッチングします。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from modules_core.bunsetu import BunsetsuSegmenter, nlp
from modules_core.cky_table import CkyTable
from modules_core.matcher import CKYMatcher
from pattern.pattern_parser import PatternParser


def build_candidate_tree(clauses: List[List[Any]], sentence: str) -> Dict[str, Any]:
    """
    トークン列から単純な右枝分かれの候補木を作る。
    各 leaf は candidate/text/xpos/upos を持つ（文節単位）。
    """
    leaves = []
    for cl in clauses:
        surface = cl[0]
        upos_list = cl[3] if len(cl) > 3 else []
        xpos_list = cl[4] if len(cl) > 4 else []
        leaves.append(
            {
                "candidate": surface,
                "text": surface,
                "xpos": xpos_list,
                "upos": upos_list,
            }
        )

    def fold(seq: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(seq) == 1:
            return seq[0]
        return {"left": seq[0], "right": fold(seq[1:])}

    tree = fold(leaves)
    tree["text"] = sentence  # cand.get("text") 用の保険
    return tree


def chunk_by_particles(doc) -> List[List[Any]]:
    """
    簡易な文節風チャンク生成。助詞・接続詞・読点などは直前トークンに吸収する。
    """
    chunks: List[List[Any]] = []
    buf_tokens: List[str] = []
    buf_upos: List[str] = []
    buf_xpos: List[str] = []
    buf_spans: List[tuple[int, int]] = []

    def flush():
        if not buf_tokens:
            return
        surface = "".join(buf_tokens)
        start = buf_spans[0][0] + 1
        end = buf_spans[-1][1]
        chunks.append([surface, (start, end), buf_tokens.copy(), buf_upos.copy(), buf_xpos.copy(), buf_spans.copy()])
        buf_tokens.clear()
        buf_upos.clear()
        buf_xpos.clear()
        buf_spans.clear()

    for tok in doc:
        text = tok.text
        is_particle_like = tok.pos_ in {"ADP", "SCONJ", "PART"} or text in {"、", "。"}
        if buf_tokens and is_particle_like:
            buf_tokens.append(text)
            buf_upos.append(tok.pos_)
            buf_xpos.append(tok.tag_)
            buf_spans.append((tok.idx, tok.idx + len(text)))
        else:
            flush()
            buf_tokens.append(text)
            buf_upos.append(tok.pos_)
            buf_xpos.append(tok.tag_)
            buf_spans.append((tok.idx, tok.idx + len(text)))
    flush()
    return chunks


def run(sentence: str, pattern: str, verbose: bool = False) -> None:
    seg = BunsetsuSegmenter()
    try:
        clauses = seg.segment(sentence)
    except Exception:
        clauses = None

    if not clauses:
        doc = nlp(sentence)
        clauses = chunk_by_particles(doc)
    cky = CkyTable.create_initializing_cky_table(clauses)

    span = len(clauses)
    candidate_tree = build_candidate_tree(clauses, sentence)
    cky[1][span] = {"candidates": [candidate_tree]}

    ast = PatternParser().parse(pattern)
    matcher = CKYMatcher(ast, verbose=verbose)

    results = matcher.match_table(cky)
    if not results:
        print("マッチなし")
        return

    print(f"マッチ件数: {len(results)}")
    for r in results:
        print(f"[span {r.i},{r.j}] {r.variable_mapping}")


def main():
    ap = argparse.ArgumentParser(description="CKYMatcher 統合テスト (文＋パターン)")
    ap.add_argument("--sentence", default="会社の仕事を太郎と花子が担当する。", help="解析対象の文")
    ap.add_argument("--pattern", default="[X0]を[G{3,3}][Y1]する",  help='パターン文字列 (例: "[X1]は[Y2]を買う")')
    ap.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    args = ap.parse_args()

    run(args.sentence, args.pattern, args.verbose)


if __name__ == "__main__":
    main()

# python matcher_integration_runner.py --sentence="リンゴとみかんを購入する太郎と花子を監視する子供がいる。" --pattern="[X0]&[X1]を[Y1]する[X2]&[X3]を"
