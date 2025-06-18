import io
import json
from contextlib import redirect_stdout

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from cky_table import CkyTable
from bert_modules import CKYAnalyzer
from matcher import CKYMatcher, MatchResult
from pattern_parser import PatternParser

# post_process copied from main.py to avoid side effects

def post_process(results):
    def finalize(info):
        data = [[ident, list(tokens), kind, pos, set(ps)] for ident, tokens, kind, pos, ps in info]
        for row in data:
            tokens = row[1]
            while tokens and tokens[-1] in {"。", "、"}:
                tokens.pop()
        for idx, (ident, tokens, kind, _, _) in enumerate(data):
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
        merged = []
        i = 0
        while i < len(data):
            ident, tokens, kind, pos, ps = data[i]
            if kind == "modifier":
                mod_tokens = tokens[:]
                mod_id = ident
                i += 1
                while i < len(data) and data[i][2] == "modifier":
                    mod_id += data[i][0]
                    mod_tokens.extend(data[i][1])
                    i += 1
                if i < len(data) and data[i][2] == "variable":
                    v_ident, v_tokens, _, v_pos, v_ps = data[i]
                    ident = mod_id + v_ident
                    tokens = mod_tokens + v_tokens
                    kind = "variable"
                    pos = v_pos
                    ps |= v_ps
                    i += 1
                else:
                    ident = mod_id
                    tokens = mod_tokens
                merged.append([ident, tokens, kind, pos, ps])
                continue
            merged.append([ident, tokens, kind, pos, ps])
            i += 1
        final_nodes = []
        i = 0
        while i < len(merged):
            ident, tokens, kind, pos, ps = merged[i]
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
            final_nodes.append([ident, tokens, kind, pos, ps])
            i += 1
        result = []
        for ident, tokens, kind, pos, ps in final_nodes:
            if kind != "variable":
                continue
            text = "".join(tokens)
            if pos and "サ変" in pos and not text.endswith("する"):
                text += "する"
            result.append((ident, text, ps))
        return result
    def expand(mapping):
        y_vars = [(k, v, p) for k, v, p in mapping if k.startswith("Y")]
        if len(y_vars) <= 1:
            return [mapping]
        base = [(k, v, p) for k, v, p in mapping if not k.startswith("Y")]
        return [base + [y] for y in y_vars]
    if not results:
        return []
    best = {}
    for r in results:
        start = r.start or 0
        for mapping in expand(finalize(r.node_info or [])):
            sym_key = tuple(k for k, _, _ in mapping)
            def weight(ps):
                return 2 if "NOUN" in ps else (1 if "VERB" in ps or "AUX" in ps else 0)
            score = sum(weight(ps) for _, _, ps in mapping)
            if any(("NOUN" not in ps and "VERB" not in ps) for _, _, ps in mapping):
                continue
            cur = best.get(sym_key)
            if cur is None or score > cur[1] or (score == cur[1] and r.i > cur[2][0]):
                best[sym_key] = (start, score, (r.i, r.j, mapping))
    return [val[2] for val in sorted(best.values(), key=lambda x: (x[0], -x[1]))]

def build_table(chunks):
    idx = 1
    clauses = []
    for text, tokens, pos in chunks:
        spans = [[idx + i, idx + i] for i in range(len(tokens))]
        idx += len(tokens)
        span = [spans[0][0], spans[-1][1]]
        clauses.append([text, span, tokens, pos, spans])
    table = CkyTable.create_initializing_cky_table(clauses)
    return table

TEST_CASES = [
    (
        "効果的な戦略を策定および実行する。",
        [
            ("効果的な", ["効果的", "な"], ["ADJ", "AUX"]),
            ("戦略を", ["戦略", "を"], ["NOUN", "ADP"]),
            ("策定および", ["策定", "および"], ["NOUN", "CCONJ"]),
            ("実行する。", ["実行", "する", "。"], ["VERB", "AUX", "PUNCT"]),
        ],
        "[*1X1]を[Y1]&[Y2]する",
        [
            {"*1X1": "効果的な戦略", "Y1": "策定する"},
            {"*1X1": "効果的な戦略", "Y2": "実行する"},
        ],
    ),
    (
        "迅速な対応を検討して実施する。",
        [
            ("迅速な", ["迅速", "な"], ["ADJ", "AUX"]),
            ("対応を", ["対応", "を"], ["NOUN", "ADP"]),
            ("検討して", ["検討", "して"], ["NOUN", "AUX"]),
            ("実施する。", ["実施", "する", "。"], ["VERB", "AUX", "PUNCT"]),
        ],
        "[*1X1]を[Y1]して[Y2]する",
        [
            {"*1X1": "迅速な対応", "Y1": "検討して"},
            {"*1X1": "迅速な対応", "Y2": "実施する"},
        ],
    ),
    (
        "詳細な計画を立案し、遂行する。",
        [
            ("詳細な", ["詳細", "な"], ["ADJ", "AUX"]),
            ("計画を", ["計画", "を"], ["NOUN", "ADP"]),
            ("立案し、", ["立案", "し", "、"], ["VERB", "AUX", "PUNCT"]),
            ("遂行する。", ["遂行", "する", "。"], ["VERB", "AUX", "PUNCT"]),
        ],
        "[*1X1]を[Y1]し、[Y2]する",
        [
            {"*1X1": "詳細な計画", "Y1": "立案し"},
            {"*1X1": "詳細な計画", "Y2": "遂行する"},
        ],
    ),
    (
        "基本的な概念を理解して応用する。",
        [
            ("基本的な", ["基本的", "な"], ["ADJ", "AUX"]),
            ("概念を", ["概念", "を"], ["NOUN", "ADP"]),
            ("理解して", ["理解", "して"], ["VERB", "AUX"]),
            ("応用する。", ["応用", "する", "。"], ["VERB", "AUX", "PUNCT"]),
        ],
        "[*1X1]を[Y1]して[Y2]する",
        [
            {"*1X1": "基本的な概念", "Y1": "理解して"},
            {"*1X1": "基本的な概念", "Y2": "応用する"},
        ],
    ),
]


def run_case(sentence, chunks, pattern):
    table = build_table(chunks)
    initial_buf = io.StringIO()
    with redirect_stdout(initial_buf):
        CkyTable.display_simple_cky_table(table)
    analyzer = CKYAnalyzer()
    analyzed = analyzer.analyze_cky_table(table)
    analyzed_buf = io.StringIO()
    with redirect_stdout(analyzed_buf):
        CkyTable.display_simple_cky_table(analyzed)
    ast = PatternParser().parse(pattern)
    matcher = CKYMatcher(ast)
    results = matcher.match_table(analyzed)
    processed = post_process(results)
    return {
        "sentence": sentence,
        "pattern": pattern,
        "initial_table": initial_buf.getvalue(),
        "analyzed_table": analyzed_buf.getvalue(),
        "results": [ {k: v for k, v, _ in mapping} for _, _, mapping in processed ],
    }


def test_full_pipeline(tmp_path):
    out = []
    for sentence, chunks, pattern, expected in TEST_CASES:
        info = run_case(sentence, chunks, pattern)
        out.append(info)
        assert len(info["results"]) == len(expected)
        for exp in expected:
            assert exp in info["results"], f"missing {exp}"
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    # ensure file saved
    assert result_file.exists()
