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

        # --- cleanup punctuation ---
        for row in data:
            tokens = row[1]
            while tokens and tokens[-1] in {"。", "、"}:
                tokens.pop()

        # --- detach literal tokens from variables ---
        for idx, (ident, tokens, kind, _) in enumerate(data):
            if idx > 0:
                prev = data[idx - 1]
                if ident == "を" and prev[1] and prev[1][-1] == "を":
                    tokens.append(prev[1].pop())
                if ident == "&" and prev[1] and prev[1][-1] in {"および", "及び"}:
                    tokens.append(prev[1].pop())
                if ident == "する" and prev[1] and prev[1][-1] == "する":
                    tokens.append(prev[1].pop())
            if kind == "variable" and (idx + 1 >= len(data) or data[idx + 1][0] != "&"):
                while tokens and tokens[-1] in {"および", "及び"}:
                    tokens.pop()

        # --- merge modifier nodes with following variable ---
        merged: list[list] = []
        i = 0
        while i < len(data):
            ident, tokens, kind, pos = data[i]
            if kind == "modifier":
                mod_tokens = tokens[:]
                mod_id = ident
                i += 1
                while i < len(data) and data[i][2] == "modifier":
                    mod_id += data[i][0]
                    mod_tokens.extend(data[i][1])
                    i += 1
                if i < len(data) and data[i][2] == "variable":
                    v_ident, v_tokens, _, v_pos = data[i]
                    ident = mod_id + v_ident
                    tokens = mod_tokens + v_tokens
                    kind = "variable"
                    pos = v_pos
                    i += 1
                else:
                    ident = mod_id
                    tokens = mod_tokens
                merged.append([ident, tokens, kind, pos])
                continue
            merged.append([ident, tokens, kind, pos])
            i += 1

        # --- absorb trailing "する" literals into preceding variables ---
        final_nodes: list[list] = []
        i = 0
        while i < len(merged):
            ident, tokens, kind, pos = merged[i]
            if ident == "する" and final_nodes:
                added = tokens or ["する"]
                idx = len(final_nodes) - 1
                while idx >= 0:
                    node = final_nodes[idx]
                    if node[2] == "variable":
                        node[1].extend(added)
                        idx -= 1
                        if idx >= 0 and final_nodes[idx][0] == "&":
                            idx -= 1
                            continue
                        break
                    elif node[0] == "&":
                        idx -= 1
                    else:
                        break
                i += 1
                continue
            final_nodes.append([ident, tokens, kind, pos])
            i += 1

        result = []
        for ident, tokens, kind, pos in final_nodes:
            if kind != "variable":
                continue
            text = "".join(tokens)
            if pos and "サ変" in pos and not text.endswith("する"):
                text += "する"
            result.append((ident, text))
        return result

    def expand(mapping: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
        y_vars = [(k, v) for k, v in mapping if k.startswith("Y")]
        if len(y_vars) <= 1:
            return [mapping]
        base = [(k, v) for k, v in mapping if not k.startswith("Y")]
        return [base + [y] for y in y_vars]

    dedup = []
    seen = set()
    for r in results:
        for mapping in expand(finalize(r.node_info or [])):
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
