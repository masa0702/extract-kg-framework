# test_llmjp_dual.py
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import requests
from tqdm import tqdm


def post_chat(url_base: str, api_key: str, model: str, max_tokens: int, tag: str, timeout: int) -> Dict[str, Any]:
    url_base = url_base.rstrip("/")
    url = f"{url_base}/chat/completions"

    # GPUをしっかり動かすために、長めに生成させる
    prompt = (
        "あなたは日本語で文章生成を行うモデルです。\n"
        "以下の制約に従って、できるだけ長い文章を生成してください。\n"
        "制約:\n"
        "1) 箇条書きは禁止\n"
        "2) 1文は長くしすぎない\n"
        "3) 内容は『知識グラフ抽出の研究計画』について\n"
        "4) 途中で終わらず、途切れずに文章を続ける\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    t1 = time.time()

    text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    return {
        "tag": tag,
        "sec": t1 - t0,
        "chars": len(text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cky-url", default="http://llmjp:8000/v1", help="CKY用 llm-jp の /v1 までのURL")
    ap.add_argument("--onto-url", default="http://llmjp_onto:8000/v1", help="整合検証用 llm-jp の /v1 までのURL")
    ap.add_argument("--cky-model", default="llmjp-13b", help="CKY用 served model name")
    ap.add_argument("--onto-model", default="llmjp-13b-onto", help="整合検証用 served model name")
    ap.add_argument("--api-key", default="local-token", help="vLLM の API key")
    ap.add_argument("--rounds", type=int, default=3, help="同時投げを何回繰り返すか")
    ap.add_argument("--max-tokens", type=int, default=1024, help="各リクエストの生成トークン数")
    ap.add_argument("--timeout", type=int, default=600, help="HTTPタイムアウト秒")
    args = ap.parse_args()

    print("=== dual llm-jp concurrency test ===")
    print(f"CKY  : {args.cky_url}  model={args.cky_model}")
    print(f"ONTO : {args.onto_url} model={args.onto_model}")
    print(f"rounds={args.rounds} max_tokens={args.max_tokens}")
    print("nvidia-smi は別ターミナルで:  nvidia-smi -l 1\n")

    # 1ラウンドにつき「CKY用1発 + ONTO用1発」を“同時に”投げる
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i in tqdm(range(1, args.rounds + 1), desc="round"):
            futs = [
                ex.submit(
                    post_chat,
                    args.cky_url,
                    args.api_key,
                    args.cky_model,
                    args.max_tokens,
                    f"CKY(r{i})",
                    args.timeout,
                ),
                ex.submit(
                    post_chat,
                    args.onto_url,
                    args.api_key,
                    args.onto_model,
                    args.max_tokens,
                    f"ONTO(r{i})",
                    args.timeout,
                ),
            ]

            for fut in as_completed(futs):
                res = fut.result()
                print(f"{res['tag']:10s}  time={res['sec']:.2f}s  chars={res['chars']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
