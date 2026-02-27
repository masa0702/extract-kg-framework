import os
import sys

# src モジュールへのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import modules_core.ontology_verify as ov


class _StubLLMJPClient:
    def __init__(self) -> None:
        self.calls = []
        self.model = "stub-model"
        self.base_urls = ["http://stub/v1"]
        self.last_base_url = None
        self.last_url = None

    def chat_json(self, messages, json_schema=None, schema_name=None, model=None, max_tokens=0, temperature=0.0):
        self.last_base_url = self.base_urls[0]
        self.last_url = self.base_urls[0] + "/chat/completions"
        self.calls.append(
            {
                "prompt": (messages or [{}])[0].get("content", ""),
                "model": model,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
        )
        return {"verdict": 1}


def test_cache_key_includes_temperature_and_max_tokens(monkeypatch):
    stub = _StubLLMJPClient()
    monkeypatch.setattr(ov, "get_llmjp_http_for", lambda _kind: stub)

    judge = ov.OntologyJudgeLLMJP(ov.OntologyJudgeConfig(cache_size=32, max_tokens=64, temperature=0.0))
    prompt = "hello"

    assert judge.judge_prompt(prompt, temperature=0.0, max_tokens=64) == 1
    assert len(stub.calls) == 1

    # Same params -> cache hit
    assert judge.judge_prompt(prompt, temperature=0.0, max_tokens=64) == 1
    assert len(stub.calls) == 1

    # Different temperature -> must miss cache
    assert judge.judge_prompt(prompt, temperature=0.2, max_tokens=64) == 1
    assert len(stub.calls) == 2

    # Different max_tokens -> must miss cache
    assert judge.judge_prompt(prompt, temperature=0.0, max_tokens=96) == 1
    assert len(stub.calls) == 3

