# bunsetsu_segmenter.py
from __future__ import annotations
from typing import List, Dict, Any
import spacy, ginza, yaml, pathlib

# ----------------- 設定読み込み -----------------
HERE = pathlib.Path(__file__).resolve().parent
DICT_PATH = HERE / "parallel_connectives.yml"
with DICT_PATH.open(encoding="utf-8") as f:
    CONNECTIVES_PARALLEL: set[str] = set(yaml.safe_load(f))

# spaCy / GiNZA
nlp = spacy.load("ja_ginza_bert_large")

# ------------ ① 単独文節→前に吸収 --------------
def _is_parallel_connective_span(span) -> bool:
    return (
        all(tok.pos_ == "CCONJ" and tok.tag_ == "接続詞" for tok in span)
        and span[0].text in CONNECTIVES_PARALLEL
    )

def _merge_connectives(spans):
    merged = []
    for sp in spans:
        if _is_parallel_connective_span(sp) and merged:
            merged[-1] = sp.doc[merged[-1].start : sp.end]  # 前 span を拡張
        else:
            merged.append(sp)
    return merged
# -------------------------------------------------
# 名詞(及びサ変名詞) 判定
def _is_nominal(tok) -> bool:
    if tok.pos_ in {"NOUN", "PROPN"}:
        return True
    # サ変可能動詞（GiNZA では VERB）も名詞列とみなす
    return tok.pos_ == "VERB" and "サ変可能" in tok.tag_

# -------------------------------------------------

# ------------ ② 同一文節内で再分割 --------------
# span 内で繰り返し分割するバージョン
def _split_span_by_connectors(span, doc):
    """
    接続語が複数あってもすべてで分割し，順序通りに Spans を返す
    例: [整理・管理&運営] → [整理・] [管理&] [運営]
    """
    boundaries = []           # 接続語の直後インデックスを格納
    for i, tok in enumerate(span):
        if tok.text in CONNECTIVES_PARALLEL and tok.pos_ in {"NOUN", "CCONJ", "SYM"}:
            if 0 < i < len(span)-1 and _is_nominal(span[i-1]) and _is_nominal(span[i+1]):
                boundaries.append(tok.i + 1)

    if not boundaries:          # 分割不要
        return [span]

    # boundaries を使って細切れ Spans を作成
    start = span.start
    pieces = []
    for b in boundaries:
        pieces.append(doc[start:b])  # b は token.i+1 なので直後まで含む
        start = b
    pieces.append(doc[start:span.end])  # 残り
    return pieces

# ---------------- メインクラス --------------------
class BunsetsuSegmenter:
    @staticmethod
    def segment(sentence: str) -> List[List[Any]]:
        doc = nlp(sentence)

        # 1. GiNZA 基本文節
        spans = list(ginza.bunsetu_spans(doc))

        # 2. 単独接続語を前方吸収
        spans = _merge_connectives(spans)

        # 3. 文節内に残った接続語で再分割
        refined = []
        for sp in spans:
            refined.extend(_split_span_by_connectors(sp, doc))

        # 4. 出力整形
        bunsetsu_list = [
            [
                sp.text,
                (sp.start_char, sp.end_char),
                [t.text for t in sp],
                [t.pos_ for t in sp],
                [t.tag_ for t in sp],
                [(t.idx, t.idx + len(t.text)) for t in sp],
            ]
            for sp in refined
        ]
        return bunsetsu_list


    @classmethod
    def segment_sentences(cls, sentences: List[str]) -> Dict[str, List[List[Any]]]:
        """
        複数文をまとめて処理し，{sentence: bunsetsu_list} で返す。

        Parameters
        ----------
        sentences : List[str]

        Returns
        -------
        Dict[str, List[List[Any]]]
        """
        docs = nlp.pipe(sentences)
        return {
            sent: cls.segment(sent)  # type: ignore[arg-type]
            for sent, doc in zip(sentences, docs)
        }


# --- 動作確認 ---
if __name__ == "__main__":
    # test = "各製品ラインに関する組織内のコンプライアンス・プログラムの編成、開発、維持、および調整を主導する"
    # test = "信託商品および信託サービスについて、幅広い管理補助と関連業務の調整を行う"
    test = "１つ以上のコンサルティング・エンゲージメント・モジュールを主導して、給与および福利厚生（C&B）ソリューションを理解、分析、開発、および推奨する"
    seg = BunsetsuSegmenter()
    for b in seg.segment(test):
        print(b)
