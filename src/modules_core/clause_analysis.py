import spacy

from .bunsetu import BunsetsuSegmenter

seg = BunsetsuSegmenter()
nlp = spacy.load("ja_ginza")


class DependencyAnalysis():
    @staticmethod
    def analyze_sentences(sentences: list[str]) -> dict:
        """
        GiNZA 依存構造と文節（bunsetsu）情報をまとめて返す。
        文節は bunsetu.BunsetsuSegmenter.segment() で取得し、
        文字位置は従来と同じ 1 始まり・終端包含に正規化する。
        """
        docs = nlp.pipe(sentences)
        results: dict[str, dict] = {}

        for sentence, doc in zip(sentences, docs):
            # 依存情報（従来どおり）
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

            # --- 文節取得を bunsetsu.segment に変更 ---
            raw_clause_data = seg.segment(sentence)

            # --- 文字位置を 1 始まり & 終端包含 に変換 ---
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
                end_inc   = ed0                # end exclusive → inclusive
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

            # --- 結果を格納（構造は従来どおり） ---
            results[sentence] = {
                "sentence": sentence,
                "clauses": clause_data,
                "ginza_dependency": ginza_dependency,
                "dependency_table": [],  # 後続処理で埋める想定
                "results": {},
            }

        return results
