from pattern_parser import PatternParser
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher, MatchResult
import json
import os
import re

# # --- 1) 簡易 CKY 表の準備 -------------------------------------------
# # オリジナルのデータファイルが無いため、サンプル文節から CKY 表を作成する
# clauses = [
#     ["製品情報を", [1, 3], ["製品情報", "を"], ["名詞", "助詞"], [[1, 2], [3, 3]]],
#     ["管理", [4, 5], ["管理"], ["サ変"], [[4, 5]]],
#     ["する", [6, 7], ["する"], ["動詞"], [[6, 7]]],
# ]

# CkyTableObj = CkyTable()
# cky_table = CkyTableObj.create_initializing_cky_table(clauses)

# 1) CKY表の作成
CkyTable = CkyTable()
BASE_DIR = os.path.dirname(__file__)
input_json = os.path.join(BASE_DIR, "..", "data", "dependency_analysis.json")
output_json = os.path.join(BASE_DIR, "..", "data", "dependency_analysis_with_cky.json")

# CKY表を生成して保存
CkyTable.process_json_to_cky_and_save(input_json, output_json)

with open(output_json, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# すべての文に対し依存解析を付与した CKY 表を準備
analyzer = CKYAnalyzer()
tables = {}
for sentence, data in json_data.items():
    cky_table = data["dependency_table"]
    tables[sentence] = analyzer.analyze_cky_table(cky_table)

# 試すパターンを複数定義
patterns = [
    "[*1X1]を[Y1]&[Y2]する",
    "[*1X1]を[Y1]する",
]


def post_process(results: list[MatchResult]) -> list[MatchResult]:
    """Clean raw match results and expand simple parallel constructions."""

    def clean_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"[。、]+$", "", text)
        text = text.replace("および", "").replace("及び", "")
        if text.endswith("を"):
            text = text[:-1]
        return text

    dedup = []
    seen = set()
    for r in results:
        mapping = {k: clean_text(v) for k, v in r.variable_mapping.items()}

        # 展開処理: 同じ記号の1番と2番があり、2番が"する"で終わる場合
        expanded = False
        for k1 in list(mapping.keys()):
            m1 = re.match(r"([*#]?\d*[A-Za-z]+)1$", k1)
            if not m1 or k1 not in mapping:
                continue
            base = m1.group(1)
            k2 = f"{base}2"
            if k2 in mapping:
                v1 = mapping[k1]
                v2 = mapping[k2]
                if v2.endswith("する") and not v1.endswith("する"):
                    # 1番は連結、2番はベース記号に
                    res1_map = mapping.copy()
                    res1_map[k1] = v1 + "する"
                    res1_map.pop(k2)
                    key1 = tuple(sorted(res1_map.items()))
                    if key1 not in seen:
                        dedup.append(MatchResult(r.cell, r.i, r.j, res1_map))
                        seen.add(key1)

                    res2_map = mapping.copy()
                    res2_map.pop(k1)
                    res2_map.pop(k2)
                    res2_map[base] = v2
                    key2 = tuple(sorted(res2_map.items()))
                    if key2 not in seen:
                        dedup.append(MatchResult(r.cell, r.i, r.j, res2_map))
                        seen.add(key2)
                    expanded = True
                    break
        if not expanded:
            key = tuple(sorted(mapping.items()))
            if key not in seen:
                dedup.append(MatchResult(r.cell, r.i, r.j, mapping))
                seen.add(key)

    return dedup


for sentence, cky_table in tables.items():
    print(f"文: {sentence}")
    for pat in patterns:
        parser = PatternParser()
        ast = parser.parse(pat)
        matcher = CKYMatcher(ast)
        results = matcher.match_table(cky_table)
        for r in post_process(results):
            print(f"{pat}: cell({r.i},{r.j}) -> {r.variable_mapping}")
