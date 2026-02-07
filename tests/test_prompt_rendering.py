import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from modules_core.ontology_verify import load_prompt_templates, render_prompt


def test_prompt21_templates_load_and_render():
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    prompts_json = os.path.join(repo_root, "prompts", "prompts.json")
    templates = load_prompt_templates(prompts_json)

    t21 = templates.get("21")
    assert t21 is not None
    # A_prime を使わない設計に変更済み
    assert "A_prime" not in (t21.arg_names or ())

    # 欠けているキーがあっても例外にならない
    s = render_prompt(
        t21,
        {
            "relation_ja": "登場人物",
            "side": "range",
            "concept_ja": "人物",
            "argument": "ジェイソン・ボーヒーズ",
            # other_argument / context_sentence intentionally omitted
        },
    )
    assert isinstance(s, str)
    assert len(s) > 0

