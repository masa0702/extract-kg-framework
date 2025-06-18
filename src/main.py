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


def post_process(results: list[MatchResult]) -> list[tuple[int, int, list[tuple[str, str]]]]:
    """Finalize node-wise token assignments."""

    def finalize(info: list[tuple]) -> list[tuple[str, str]]:
        data = [[ident, list(tokens), kind, pos] for ident, tokens, kind, pos in info]
        # 句読点を除去
        for _, tokens, _, _ in data:
            while tokens and tokens[-1] in {"。", "、"}:
                tokens.pop()

        for idx, (ident, tokens, kind, _) in enumerate(data):
            if ident == "を" and idx > 0:
                prev = data[idx - 1]
                if prev[1] and prev[1][-1] == "を":
                    tokens.append(prev[1].pop())
            if ident == "&" and idx > 0:
                prev = data[idx - 1]
                if prev[1] and prev[1][-1] in {"および", "及び"}:
                    tokens.append(prev[1].pop())
            if ident == "する" and idx > 0:
                prev = data[idx - 1]
                if prev[1] and prev[1][-1] == "する":
                    tokens.append(prev[1].pop())
            # 単独の "および" が末尾に残った場合も削除
            if kind == "variable" and idx + 1 < len(data) and not (data[idx + 1][0] == "&"):
                if tokens and tokens[-1] in {"および", "及び"}:
                    tokens.pop()

        result = []
        for ident, tokens, kind, pos in data:
            text = "".join(tokens)
            if kind == "variable" and pos and "サ変" in pos:
                text += "する"
            result.append((ident, text))
        return result

    dedup = []
    seen = set()
    for r in results:
        mapping = finalize(r.node_info or [])
        key = (r.i, r.j, tuple(mapping))
        if key not in seen:
            seen.add(key)
            dedup.append((r.i, r.j, mapping))
    return dedup


for sentence, cky_table in tables.items():
    print(f"文: {sentence}")
    for pat in patterns:
        parser = PatternParser()
        ast = parser.parse(pat)
        matcher = CKYMatcher(ast)
        results = matcher.match_table(cky_table)
        for i, j, mapping in post_process(results):
            mapping_str = ", ".join(f"{k} = {v}" for k, v in mapping)
            print(f"{pat}: cell({i},{j}) -> {mapping_str}")
