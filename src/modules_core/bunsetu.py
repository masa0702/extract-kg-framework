# bunsetsu_segmenter.py
from __future__ import annotations
from typing import List, Dict, Any
import spacy, ginza, yaml, pathlib
import time
import logging
from functools import lru_cache

# ----------------- 設定読み込み -----------------
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_DIR = ROOT / "config"
DICT_PATH = CONFIG_DIR / "parallel_connectives.yml"
with DICT_PATH.open(encoding="utf-8") as f:
    CONNECTIVES_PARALLEL: set[str] = set(yaml.safe_load(f))

BRACKETS = [("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】")]


# spaCy / GiNZA
# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_spacy_model(name, timeout=60):
    logging.info(f"spaCyモデル「{name}」の読み込みを開始します。")
    start = time.time()
    try:
        nlp = spacy.load(name)
    except (IOError, OSError, TypeError) as e:
        logging.error(f"spaCyモデル「{name}」の読み込みに失敗しました: {e}")
        raise
    end = time.time()
    elapsed = end - start
    logging.info(f"spaCyモデル「{name}」の読み込み完了（{elapsed:.1f}秒）")
    if elapsed > timeout:
        logging.error(f"spaCyモデル「{name}」の読み込みに{timeout}秒以上かかりました（{elapsed:.1f}秒）")
        raise TimeoutError(f"spaCyモデル「{name}」の読み込みがタイムアウトしました。")
    return nlp

@lru_cache(maxsize=1)
def get_nlp():
    """
    spaCy/GiNZA モデルを遅延ロードして使い回す。
    multiprocessing(spawn) 環境では import 時ロードが子プロセスごとに走って重くなるため、
    必要になったタイミングでのみロードする。
    """
    try:
        nlp = load_spacy_model("ja_ginza_bert_large", timeout=60)
    except Exception as e:
        logging.warning(f"BERT GiNZAの読み込みに失敗したので、通常GiNZAで再試行します。: {e}")
        nlp = load_spacy_model("ja_ginza", timeout=30)
    logging.info(f"使用中のspaCyモデル: {nlp.meta['name']}")
    return nlp


# ------------- 辞書読み込み -----------------
CHUNK_PATH = CONFIG_DIR / "fixed_chunks.yml"
with CHUNK_PATH.open(encoding="utf-8") as f:
    POST_PATTERNS = [line.split() for line in yaml.safe_load(f)]

# ------------- 汎用マージ関数 ----------------
def _merge_preceding_chunk(spans, post_patterns):
    """
    pattern 直前の span ＋ pattern 部分を 1 文節に。
    末尾トークンが句読点の場合「句読点だけ別 span」でも認識。
    """
    tokens = [tok for sp in spans for tok in sp]
    tok_texts = [t.text for t in tokens]

    # span 開始インデックスをメモ
    span_starts, n = [], 0
    for sp in spans:
        span_starts.append(n)
        n += len(sp)

    merged_flags = [False]*len(tokens)
    merge_ranges = []

    for pattern in post_patterns:
        pat_len = len(pattern)
        idx = 1
        while idx <= len(tokens) - pat_len:
            window = tok_texts[idx:idx+pat_len]
            # パターン末尾が句読点なら "句読点が別 span" パターンも許容
            if window == pattern or (
                pattern[-1] in {"、", "。"} and
                tok_texts[idx:idx+pat_len-1] == pattern[:-1] and
                tok_texts[idx+pat_len-1] in {"、", "。"}
            ):
                start = idx - 1  # 直前トークンごと
                end   = idx + pat_len
                if window != pattern:
                    end += 1      # 句読点だけ別トークン分
                if not any(merged_flags[start:end]):
                    merge_ranges.append((start, end))
                    for j in range(start, end):
                        merged_flags[j] = True
                idx = end
            else:
                idx += 1

    # --- 後段は従来どおり ---
    doc = spans[0].doc
    i, result = 0, []
    while i < len(tokens):
        for s, e in merge_ranges:
            if i == s:
                result.append(doc[tokens[s].i : tokens[e-1].i+1])
                i = e
                break
        else:
            # span 単位で進む
            span_idx = max(k for k, st in enumerate(span_starts) if st <= i)
            span_end = span_starts[span_idx+1] if span_idx+1 < len(span_starts) else len(tokens)
            result.append(doc[tokens[i].i : tokens[span_end-1].i+1])
            i = span_end
    return result




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

def _merge_brackets_and_particle(spans, bracket_pairs=BRACKETS):
    """
    spans リストを走査し，開き括弧〜閉じ括弧(+直後助詞/読点) が
    複数 span に跨っていても 1 span にまとめる。
    """
    merged = []
    open_set  = {l for l, _ in bracket_pairs}
    close_map = {l: r for l, r in bracket_pairs}

    i = 0
    while i < len(spans):
        sp = spans[i]
        # span 先頭トークンが開き括弧？
        first_tok = sp[0]
        if first_tok.text in open_set:
            l_br = first_tok.text
            r_br = close_map[l_br]
            # 開き括弧 span から順に閉じ括弧 span を探す
            j = i
            found = False
            while j < len(spans):
                if any(tok.text == r_br for tok in spans[j]):
                    found = True
                    break
                j += 1
            if found:
                # i〜j まで + 直後の助詞/読点 span も取り込む
                end = j
                if end + 1 < len(spans) and all(
                        tok.pos_ in {"ADP", "PUNCT"} for tok in spans[end+1]):
                    end += 1
                merged_span = sp.doc[spans[i][0].i : spans[end][-1].i + 1]
                merged.append(merged_span)
                i = end + 1
                continue
        # 括弧開始でない → そのまま
        merged.append(sp)
        i += 1
    return merged

def _merge_trailing_comma(spans):
    """
    先頭トークンが読点「、」の span を必ず直前の span に吸収する。
    先頭が「、」で、かつ結合先が存在する場合のみマージ。
    """
    merged = []
    for sp in spans:
        if sp[0].text == "、" and merged:
            # 直前 span の開始〜現在 span の終端で新 span を作成
            prev = merged.pop()
            merged.append(sp.doc[prev.start : sp.end])
        else:
            merged.append(sp)
    return merged


import re

def _merge_alnum_katakana_spans(spans):
    """
    英数字・カタカナの連続をひとまとまりにマージする
    """
    def is_alnum_or_katakana(text):
        # 英字・数字・カタカナにマッチ
        return bool(re.match(r'^[A-Za-z0-9ァ-ンヴー・\'’]+$', text))
    merged = []
    buf = []
    for sp in spans:
        if all(is_alnum_or_katakana(t.text) for t in sp):
            buf.append(sp)
        else:
            if buf:
                # まとめてひとつのspanに
                start = buf[0].start
                end = buf[-1].end
                merged.append(sp.doc[start:end])
                buf = []
            merged.append(sp)
    if buf:
        start = buf[0].start
        end = buf[-1].end
        merged.append(sp.doc[start:end])
    return merged


ENG_RE = re.compile(r"^[A-Za-z0-9'’-]+$")  # ハイフンやアポストロフィも許容
JP_PARTICLES = {
    "が",
    "は",
    "を",
    "に",
    "へ",
    "と",
    "で",
    "も",
    "や",
    "の",
    "から",
    "まで",
    "より",
}


def _is_jp_particle_token(tok) -> bool:
    try:
        text = tok.text
        pos = tok.pos_
    except Exception:
        return False
    if pos == "ADP" and text in JP_PARTICLES:
        return True
    return False


def _is_englishish_span(sp) -> bool:
    tokens = list(sp)
    return all(ENG_RE.match(tok.text) or tok.pos_ in {"ADP", "PUNCT", "SYM"} for tok in tokens)

def _merge_consecutive_english_spans(spans):
    """
    連続する英語（英数字、記号、助詞含む）のspanをまとめて1つのspanにする。
    英語以外（日本語名詞や動詞など）が登場した時点で区切る。
    """
    merged = []
    buffer = []
    buffer_start = None
    buffer_end = None
    state = "idle"  # "idle" or "in_english"
    doc = spans[0].doc if spans else None

    for sp in spans:
        # 英語トークンまたは「英語＋助詞/記号」だけの場合
        # -> ただし、先頭は必ず英語で始まる
        tokens = list(sp)
        if (state == "idle" and ENG_RE.match(tokens[0].text)):
            # 連結開始
            buffer_start = sp.start
            buffer_end = sp.end
            buffer = [sp]
            state = "in_english"
        elif state == "in_english" and _is_englishish_span(sp):
            prev_last = list(buffer[-1])[-1] if buffer else None
            if prev_last is not None and _is_jp_particle_token(prev_last):
                merged.append(doc[buffer_start:buffer_end])
                buffer_start = sp.start
                buffer_end = sp.end
                buffer = [sp]
                state = "in_english"
            else:
                buffer_end = sp.end
                buffer.append(sp)
        else:
            if buffer:
                merged.append(doc[buffer_start:buffer_end])
                buffer = []
                state = "idle"
            merged.append(sp)
    # 最後にバッファが残っていれば出力
    if buffer:
        merged.append(doc[buffer_start:buffer_end])
    return merged


# ---------------- 空白ブリッジ（英語系） ----------------
def _merge_whitespace_bridged_entityish_spans(spans):
    """
    空白（スペース）をまたいで分割された span を、1文節として扱えるようにマージする。

    方針:
    - doc.text 上で span 間が「空白のみ」の場合、助詞境界でなければ連結する。
      （英語・日本語・数字を含む表記ゆれの分割を避ける）
    """
    if not spans:
        return spans

    doc = spans[0].doc

    merged = []
    buf = [spans[0]]

    for sp in spans[1:]:
        prev = buf[-1]
        bridge = doc.text[prev.end_char : sp.start_char]
        if bridge and bridge.isspace():
            prev_last = list(prev)[-1] if prev is not None else None
            next_first = list(sp)[0] if sp is not None and len(sp) > 0 else None
            if not _is_jp_particle_token(prev_last) and not _is_jp_particle_token(next_first):
                buf.append(sp)
                continue
        # flush
        if len(buf) >= 2:
            merged.append(doc[buf[0].start : buf[-1].end])
        else:
            merged.append(buf[0])
        buf = [sp]

    if buf:
        if len(buf) >= 2:
            merged.append(doc[buf[0].start : buf[-1].end])
        else:
            merged.append(buf[0])
    return merged


# ---------------- メインクラス --------------------
class BunsetsuSegmenter:
    @staticmethod
    def segment(sentence: str) -> List[List[Any]]:
        doc = get_nlp()(sentence)
        return BunsetsuSegmenter._segment_doc(doc)

    @staticmethod
    def _segment_doc(doc) -> List[List[Any]]:
        spans = list(ginza.bunsetu_spans(doc))             # ① GiNZA
        spans = _merge_connectives(spans)                  # ② 接続語吸収
        spans = _merge_brackets_and_particle(spans)        # ③ 括弧マージ
        spans = _merge_preceding_chunk(spans, POST_PATTERNS)  # ④ 固定句マージ
        spans = _merge_consecutive_english_spans(spans)    # ★⑤ 英語連結  ← NEW
        spans = _merge_alnum_katakana_spans(spans)         # ⑥ 英数字+カナ結合
        spans = _merge_whitespace_bridged_entityish_spans(spans)  # ⑥.5 空白ブリッジ（英語系）
        spans = _merge_trailing_comma(spans)               # ⑦ 読点吸収

        # 6. 文節内の接続語で再分割
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
        docs = get_nlp().pipe(sentences)
        return {
            sent: cls._segment_doc(doc)  # type: ignore[arg-type]
            for sent, doc in zip(sentences, docs)
        }


# ---------------- 依存解析 ----------------
class DependencyAnalysis:
    @staticmethod
    def analyze_sentences(sentences: list[str]) -> dict:
        """
        GiNZA 依存構造と文節（bunsetsu）情報をまとめて返す。
        文節は BunsetsuSegmenter.segment() で取得し、
        文字位置は 1 始まり・終端包含に正規化する。
        """
        docs = get_nlp().pipe(sentences)
        results: dict[str, dict] = {}
        segmenter = BunsetsuSegmenter()

        for sentence, doc in zip(sentences, docs):
            ginza_dependency = [
                {
                    "id": tok.i + 1,
                    "token": tok.text,
                    "lemma": tok.lemma_,
                    "upos": tok.pos_,
                    "xpos": tok.tag_,
                    "head": tok.head.i,
                    "deprel": tok.dep_,
                    "misc": tok.morph.to_dict(),
                }
                for tok in doc
            ]

            raw_clause_data = segmenter._segment_doc(doc)

            clause_data = []
            for (
                text_span,
                (st0, ed0),           # 0-based, end exclusive
                tokens,
                upos_list,
                xpos_list,
                ranges0,
            ) in raw_clause_data:
                start_inc = st0 + 1            # 1-based, inclusive
                end_inc = ed0                  # end exclusive → inclusive
                ranges_inc = [(s + 1, e) for s, e in ranges0]

                clause_data.append(
                    [
                        text_span,
                        (start_inc, end_inc),
                        tokens,
                        upos_list,
                        xpos_list,
                        ranges_inc,
                    ]
                )

            results[sentence] = {
                "sentence": sentence,
                "clauses": clause_data,
                "ginza_dependency": ginza_dependency,
                "dependency_table": [],  # 後続処理で埋める想定
                "results": {},
            }

        return results


# --- 動作確認 ---
if __name__ == "__main__":
    # test = "各製品ラインに関する組織内のコンプライアンス・プログラムの編成、開発、維持、および調整を主導する"
    test = "リンゴとみかんを購入する太郎と花子を監視する。"
    # test = "信託商品および信託サービスについて、幅広い管理補助と関連業務の調整を行う"
    # test = "Man on Horsebackはフォルカー・シュレンドルフが監督、脚本を務め、1969年1月1日に公開され、ミヒャエル・コールハースを原作としています。"
    # test = "「涼宮ハルヒの消失」は、の制作会社が京都アニメーション、脚本家が志茂文彦、監督が石原立也と武本康弘である作品です"
    seg = BunsetsuSegmenter()
    for b in seg.segment(test):
        print(b)
