import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


@dataclass(frozen=True)
class GenParams:
    max_new_tokens: int = 16
    do_sample: bool = False
    top_p: float = 0.95
    temperature: float = 0.1
    repetition_penalty: float = 1.05


class LLMJPHTTPClient:
    """
    vLLM(OpenAI互換) サーバを叩く軽量クライアント。
    - エンドポイント: {base_url}/chat/completions
    - vLLMはOpenAI互換APIに加えて、extra params をpayloadに混ぜて渡せる。 :contentReference[oaicite:3]{index=3}
    - guided_json などの構造化出力もpayloadに追加可能。 :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: str = "",
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        backoff_sec: float = 0.5,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLMJP_BASE_URL", "http://llmjp:8000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLMJP_API_KEY", "local-token")
        self.model = model or os.getenv("LLMJP_MODEL", "llmjp-13b")
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec

        self._session = requests.Session()
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_messages(self, user_prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        sp = self.system_prompt if system_prompt is None else system_prompt
        if sp:
            return [{"role": "system", "content": sp}, {"role": "user", "content": user_prompt}]
        return [{"role": "user", "content": user_prompt}]

    def _build_payload(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        params: GenParams,
        guided_json: Optional[Union[Dict[str, Any], str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # OpenAI互換 Chat Completions 形式
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(user_prompt, system_prompt),
            # OpenAI互換: max_tokens は「生成する最大トークン数」
            "max_tokens": int(params.max_new_tokens),
        }

        # サンプリング設定
        if params.do_sample:
            payload["temperature"] = float(params.temperature)
            payload["top_p"] = float(params.top_p)
        else:
            # 決定的に寄せる（vLLM側の実装にも依るが、temperature=0 は一般に安定）
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0

        # vLLM extra params（OpenAI非標準）: repetition_penalty 等をpayloadに混ぜる :contentReference[oaicite:5]{index=5}
        payload["repetition_penalty"] = float(params.repetition_penalty)

        # 構造化出力（JSON Schemaを強制する guided_json） :contentReference[oaicite:6]{index=6}
        if guided_json is not None:
            payload["guided_json"] = guided_json

        if extra:
            payload.update(extra)

        return payload

    def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(url, headers=self._headers, json=payload, timeout=self.timeout_sec)

                # vLLM/OpenAI互換サーバとしての一般的な一時エラーはリトライ対象
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

                r.raise_for_status()
                return r.json()

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_sec * (2 ** attempt))

        raise RuntimeError(f"vLLM呼び出しに失敗: {last_err}") from last_err

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[GenParams] = None,
        guided_json: Optional[Union[Dict[str, Any], str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        params = params or GenParams()
        payload = self._build_payload(prompt, system_prompt, params, guided_json=guided_json, extra=extra)
        data = self._post_with_retry("/chat/completions", payload)

        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            raise RuntimeError(f"想定外のレスポンス形式: {data}") from e

    def generate_many(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str] = None,
        params: Optional[GenParams] = None,
        guided_json: Optional[Union[Dict[str, Any], str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        max_workers: int = 8,
    ) -> List[str]:
        """
        OpenAI互換APIはバッチ入力を標準化していないため、クライアント側で並列POSTする。
        vLLM側は連続バッチング等で効率化されやすい。 :contentReference[oaicite:7]{index=7}
        """
        params = params or GenParams()
        results: List[Optional[str]] = [None] * len(prompts)

        def _one(idx: int, p: str) -> str:
            return self.generate(p, system_prompt=system_prompt, params=params, guided_json=guided_json, extra=extra)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_one, i, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()

        # mypy対策
        return [r if r is not None else "" for r in results]


# 使い回し用（mainプロセス内でHTTPセッションを共有したい場合）
_SINGLETON: Optional[LLMJPHTTPClient] = None


def get_llmjp_http(system_prompt: str = "") -> LLMJPHTTPClient:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMJPHTTPClient(system_prompt=system_prompt)
    return _SINGLETON
