from llm.llmjp_client import get_llmjp_http, GenParams

def main() -> None:
    client = get_llmjp_http(system_prompt="あなたは以下の平常文と疑問文が両方成立していればYes、成立していなければNoだけで答えるAIです。")
    out = client.generate(
        "平常：映画はヒトです。　疑問：映画はヒトですか？　両方の文が成立していればYes、成立していなければNoだけで答えてください。",
        params=GenParams(max_new_tokens=1024, do_sample=False),
    )
    print(out)

if __name__ == "__main__":
    main()
