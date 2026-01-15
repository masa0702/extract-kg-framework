from llm.llmjp_client import get_llmjp_http, GenParams

def main() -> None:
    client = get_llmjp_http(system_prompt="あなたは簡潔に答える。")
    out = client.generate(
        "りんごとみかんは何？",
        params=GenParams(max_new_tokens=1024, do_sample=False),
    )
    print(out)

if __name__ == "__main__":
    main()
