import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
import json


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
    - vLLMはOpenAI互換APIに加えて、extra params をpayloadに混ぜて渡せる。
    - guided_json / response_format(json_schema) などの構造化出力もpayloadに追加可能。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        base_urls: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: str = "",
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        backoff_sec: float = 0.5,
    ) -> None:
        # base_urls を渡すと、複数の vLLM サーバへラウンドロビン＋リトライ時フェイルオーバーする。
        # 互換性のため base_url（単一）も残す。
        if base_urls is not None:
            urls = [u.strip().rstrip("/") for u in base_urls if u and u.strip()]
        else:
            urls = []

        if not urls:
            urls = [(base_url or os.getenv("LLMJP_BASE_URL", "http://llmjp:8000/v1")).rstrip("/")]

        self.base_urls = urls
        self.api_key = api_key or os.getenv("LLMJP_API_KEY", "local-token")
        self.model = model or os.getenv("LLMJP_MODEL", "llmjp-13b")
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec

        self._rr_lock = threading.Lock()
        self._rr_i = 0

        self._session = requests.Session()
        self._session.trust_env = False  # 環境変数 *_proxy を参照しない（内部通信をプロキシ経由にしない）
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Dict[str, Any],
        schema_name: str,
        model: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        vLLM OpenAI互換 /v1/chat/completions を使い、
        response_format=json_schema で構造化出力を強制して JSON を返す。
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            },
        }

        data = self._post_with_retry("/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def _pick_base_url(self, *, attempt: int) -> str:
        # attempt を混ぜて、リトライ時に別サーバへ移る（サーバ落ち/過負荷時の回復を狙う）
        n = len(self.base_urls)
        if n <= 1:
            return self.base_urls[0]
        with self._rr_lock:
            i = self._rr_i
            self._rr_i = (self._rr_i + 1) % n
        return self.base_urls[(i + attempt) % n]

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

        # vLLM extra params（OpenAI非標準）: repetition_penalty 等をpayloadに混ぜる
        payload["repetition_penalty"] = float(params.repetition_penalty)

        # 構造化出力（JSON Schemaを強制する guided_json）
        if guided_json is not None:
            payload["guided_json"] = guided_json

        if extra:
            payload.update(extra)

        return payload

    def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                base = self._pick_base_url(attempt=attempt)
                url = f"{base}{path}"
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
        vLLM側は連続バッチング等で効率化されやすい。
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
_SINGLETONS: Dict[tuple, LLMJPHTTPClient] = {}


def _split_base_urls(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    # カンマ区切りを想定（"http://.../v1,http://.../v1"）
    urls = [x.strip() for x in s.split(",") if x.strip()]
    return urls or None


def get_llmjp_http(
    *,
    base_url: Optional[str] = None,
    base_urls: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: str = "",
    timeout_sec: float = 60.0,
    max_retries: int = 3,
    backoff_sec: float = 0.5,
) -> LLMJPHTTPClient:
    """
    設定（base_url/model/timeout等）ごとに LLMJPHTTPClient を使い回す。
    多回呼び出しで Session を毎回作らないための最適化。
    """
    # base_urls が指定されていなければ、環境変数 LLMJP_BASE_URLS を参照
    env_timeout = os.getenv("LLMJP_TIMEOUT_SEC")
    env_retries = os.getenv("LLMJP_MAX_RETRIES")
    env_backoff = os.getenv("LLMJP_BACKOFF_SEC")
    if env_timeout:
        try:
            timeout_sec = float(env_timeout)
        except Exception:
            pass
    if env_retries:
        try:
            max_retries = int(env_retries)
        except Exception:
            pass
    if env_backoff:
        try:
            backoff_sec = float(env_backoff)
        except Exception:
            pass
    env_urls = _split_base_urls(os.getenv("LLMJP_BASE_URLS"))
    resolved_urls = list(base_urls) if base_urls is not None else (env_urls or None)
    resolved_url0 = (base_url or os.getenv("LLMJP_BASE_URL", "http://llmjp:8000/v1")).rstrip("/")

    key = (
        tuple([u.rstrip("/") for u in resolved_urls]) if resolved_urls else (resolved_url0,),
        api_key or os.getenv("LLMJP_API_KEY", "local-token"),
        model or os.getenv("LLMJP_MODEL", "llmjp-13b"),
        system_prompt,
        float(timeout_sec),
        int(max_retries),
        float(backoff_sec),
    )

    c = _SINGLETONS.get(key)
    if c is None:
        c = LLMJPHTTPClient(
            base_url=base_url,
            base_urls=resolved_urls,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        )
        _SINGLETONS[key] = c
    return c


def get_llmjp_http_for(
    use: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: str = "",
    timeout_sec: float = 60.0,
    max_retries: int = 3,
    backoff_sec: float = 0.5,
) -> LLMJPHTTPClient:
    """用途別にエンドポイントを切り替える。

    想定する use:
      - "cky"  : CKY内の並列検証用
      - "onto" : オントロジー整合性検証（Paraphrase等）用

    環境変数:
      - LLMJP_CKY_BASE_URL / LLMJP_CKY_BASE_URLS
      - LLMJP_ONTO_BASE_URL / LLMJP_ONTO_BASE_URLS
    """
    u = (use or "").strip().lower()
    if u == "cky":
        bu = os.getenv("LLMJP_CKY_BASE_URL")
        bus = _split_base_urls(os.getenv("LLMJP_CKY_BASE_URLS"))
    elif u == "onto":
        bu = os.getenv("LLMJP_ONTO_BASE_URL")
        bus = _split_base_urls(os.getenv("LLMJP_ONTO_BASE_URLS"))
    else:
        bu = os.getenv("LLMJP_BASE_URL")
        bus = _split_base_urls(os.getenv("LLMJP_BASE_URLS"))

    return get_llmjp_http(
        base_url=bu,
        base_urls=bus,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
