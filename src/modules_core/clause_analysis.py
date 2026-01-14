import pandas as pd
import json
import spacy
import json
import csv
import re

from itertools import combinations
from spacy import displacy
from .utils import MyUtility
from .bunsetu import BunsetsuSegmenter

seg = BunsetsuSegmenter()
nlp = spacy.load("ja_ginza")
myutil = MyUtility()


class DependencyAnalysis():
    @staticmethod
    def clause_search(json_data, sentence):
        """
        JSON データから文節を抽出し、文字列スパンを計算する関数。

        Args:
            json_data (list): 文節情報を含む辞書のリスト。各辞書は「deprel」「upos」「token」「id」「head」などのキーを持つ。
            sentence (str): 文全体の文字列。

        Returns:
            list: 抽出された文節のリスト。各文節は形式 [text, (start_char, end_char), tokens, upos_list, value_ranges]。
        """
        clause_list = []
        clause_text = ""
        clause_value_st = 0
        clause_value_ed = 0
        clause_element = []
        json_len = len(json_data)
        clause_first_flag = True
        char_pointer = 1  # 1から開始する文字列ポインタ

        for value, data in enumerate(json_data):
            token_text = data["token"]
            token_len = len(token_text)
            token_start = char_pointer  # 現在のポインタ位置を開始位置とする
            token_end = token_start + token_len - 1  # 終了位置を計算
            char_pointer = token_end + 1  # ポインタを次の位置に更新

            target_deprel = data["deprel"]
            target_upos = data["upos"]
            target_xpos = data["xpos"]
            print(target_xpos)

            if value + 1 < json_len:
                next_sentence = json_data[value + 1]
                next_upos = next_sentence["upos"]
                next_deprel = next_sentence["deprel"]
                next_text = next_sentence["token"]
            else:
                next_upos = next_deprel = next_text = None

            # 文頭の処理
            if clause_first_flag:
                clause_value_st = token_start
                clause_text += token_text
                clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                if value + 1 == json_len:
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    xpos_list = [element[3] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                    clause_element = []
                    clause_text = ""
                elif target_upos == "VERB" and target_deprel == "fixed":
                    if next_upos == "SCONJ" and next_deprel == "fixed":
                        clause_first_flag = False
                    elif myutil.check_comma(next_text):
                        clause_first_flag = False
                    elif myutil.check_period(next_text):
                        clause_first_flag = False
                    else:
                        clause_value_ed = token_end
                        tokens = [element[0] for element in clause_element]
                        xpos_list = [element[3] for element in clause_element]
                        value_ranges = [element[1] for element in clause_element]
                        clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                        clause_element = []
                        clause_first_flag = True
                        clause_text = ""
                else:
                    clause_first_flag = False

            # 文末の処理
            elif myutil.check_period(token_text):
                clause_text += token_text
                clause_value_ed = token_end
                clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                xpos_list = [element[3] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 句点の処理
            elif myutil.check_comma(token_text):
                clause_text += token_text
                clause_value_ed = token_end
                clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                xpos_list = [element[3] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 記号の処理
            elif target_upos == "DET" and target_deprel == "det":
                clause_text += token_text
                clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                clause_value_ed = token_end
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                xpos_list = [element[3] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 接続詞の処理
            elif target_upos == "CCONJ" and (target_deprel == "cc" or target_deprel == "advmod"):
                if next_upos == "ADP" and next_deprel == "fixed":
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    xpos_list = [element[3] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # 前置詞の処理
            elif target_upos == "ADP" and target_deprel == "case":
                if next_upos == "ADP" and next_deprel == "case":
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    xpos_list = [element[3] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # 動詞や補助動詞の処理
            elif (target_upos in ["VERB", "SCONJ", "AUX"] and target_deprel in ["fixed", "mark"]) or \
                (target_upos == "AUX" and target_deprel in ["aux", "cop"]):
                if next_upos in ["SCONJ", "AUX"] and next_deprel in ["mark", "aux"]:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                elif myutil.check_period(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    xpos_list = [element[3] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, xpos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # その他の処理
            else:
                clause_text += token_text
                clause_element.append((token_text, (token_start, token_end), target_upos, target_xpos))
                clause_first_flag = False

        return clause_list


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


def main():
    sentences = [
        "技術的な製品情報を記述および管理する。"
    ]
    depana = DependencyAnalysis()
    output_filename = "../data/dependency_analysis.json"
    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    new_sentences = [s for s in sentences if s not in data]

    if new_sentences:
        new_results = depana.analyze_sentences(new_sentences)
        data.update(new_results)
        myutil.save_json_from_file(data, output_filename)

    print(f"全ての文の解析結果を {output_filename} に保存しました。")

if __name__ == "__main__":
    main()
