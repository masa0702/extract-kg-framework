import pandas as pd
import json
import spacy
import json
import csv
import re

from itertools import combinations
from spacy import displacy
from utils import MyUtility

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
                clause_element.append((token_text, (token_start, token_end), target_upos))
                if value + 1 == json_len:
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
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
                        upos_list = [element[2] for element in clause_element]
                        value_ranges = [element[1] for element in clause_element]
                        clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                        clause_element = []
                        clause_first_flag = True
                        clause_text = ""
                else:
                    clause_first_flag = False

            # 文末の処理
            elif myutil.check_period(token_text):
                clause_text += token_text
                clause_value_ed = token_end
                clause_element.append((token_text, (token_start, token_end), target_upos))
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 句点の処理
            elif myutil.check_comma(token_text):
                clause_text += token_text
                clause_value_ed = token_end
                clause_element.append((token_text, (token_start, token_end), target_upos))
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 記号の処理
            elif target_upos == "DET" and target_deprel == "det":
                clause_text += token_text
                clause_element.append((token_text, (token_start, token_end), target_upos))
                clause_value_ed = token_end
                tokens = [element[0] for element in clause_element]
                upos_list = [element[2] for element in clause_element]
                value_ranges = [element[1] for element in clause_element]
                clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                clause_element = []
                clause_first_flag = True
                clause_text = ""

            # 接続詞の処理
            elif target_upos == "CCONJ" and (target_deprel == "cc" or target_deprel == "advmod"):
                if next_upos == "ADP" and next_deprel == "fixed":
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # 前置詞の処理
            elif target_upos == "ADP" and target_deprel == "case":
                if next_upos == "ADP" and next_deprel == "case":
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # 動詞や補助動詞の処理
            elif (target_upos in ["VERB", "SCONJ", "AUX"] and target_deprel in ["fixed", "mark"]) or \
                (target_upos == "AUX" and target_deprel in ["aux", "cop"]):
                if next_upos in ["SCONJ", "AUX"] and next_deprel in ["mark", "aux"]:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                elif myutil.check_comma(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                elif myutil.check_period(next_text):
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_first_flag = False
                else:
                    clause_text += token_text
                    clause_element.append((token_text, (token_start, token_end), target_upos))
                    clause_value_ed = token_end
                    tokens = [element[0] for element in clause_element]
                    upos_list = [element[2] for element in clause_element]
                    value_ranges = [element[1] for element in clause_element]
                    clause_list.append([clause_text, (clause_value_st, clause_value_ed), tokens, upos_list, value_ranges])
                    clause_element = []
                    clause_first_flag = True
                    clause_text = ""

            # その他の処理
            else:
                clause_text += token_text
                clause_element.append((token_text, (token_start, token_end), target_upos))
                clause_first_flag = False

        return clause_list


    @staticmethod
    def analyze_sentences(sentences: str) -> dict:
        # 文の解析を一括で処理
        docs = nlp.pipe(sentences)

        results = {}
        for sentence, doc in zip(sentences, docs):
            ginza_dependency = []
            for token in doc:
                ginza_dependency.append({
                    "id": token.i+1,
                    "token": token.text,
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "head": token.head.i,
                    "deprel": token.dep_,
                    "misc": token.morph.to_dict(),
                })
            # 文節解析
            clause_data = DependencyAnalysis.clause_search(ginza_dependency, sentence)

            # 結果のJSON構造を作成
            results[sentence] = {
                "sentence": sentence,
                "clauses": clause_data,
                "ginza_dependency": ginza_dependency,
                "dependency_table": [],
                "results": {}
            }
        return results

def main():
    sentences = [
        "技術的な製品情報を記述および管理する。"
    ]
    depana = DependencyAnalysis()
    output_filename = "dependency_analysis.json"
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