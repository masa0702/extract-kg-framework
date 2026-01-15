# parallel_judge.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from llm.llmjp_client import get_llmjp_http


# =========================
# 並列判定（LLM-jp / vLLM）
# =========================
#
# CKYセル内で高頻度に呼ぶ前提で、以下を重視:
# - プロンプト短縮（判定タスクに限定）
# - 出力を boolean に統一（JSON Schema 強制）
# - max_tokens 最小化
# - LRU キャッシュ
# - 低オーバーヘッドの統計（calls / misses / failures / latency）
#
# 返り値は常に bool（True/False）。None は返さない。
# 失敗時は False に倒す（strict=True の場合のみ例外送出）。


# vLLM の structured outputs / json schema で強制する最小スキーマ
_PARALLEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "judge_result": {"type": "boolean"},
    },
    "required": ["judge_result"],
    "additionalProperties": False,
}

_SYSTEM = "あなたは日本語の文法・意味に精通した専門家です。"

# 出力は boolean のみ（"True"/"False" 文字列禁止）
_USER_TMPL = """\
        目的：
        対象文における並列要素について次の2点を厳密に確認し、「True」または「False」をjson形式で出力してください。

        【入力情報】
        - 対象文: 「{sentence}」
        - 並列要素: {elements_str}（例: 「"A"」,「"B"」,「"C"」）
        
        【判定基準】
        1. 類似性の確認：
        - 各並列要素の主要語（中心語）が**同じ品詞**であること（例：全て名詞、または全て動詞）。
        - 句構造もできる限り揃っていること（例：全て「名詞＋助詞」の形など）。

        2. 可換性の確認：
        - 並列要素の順序を入れ替えた場合でも、**日本語として自然な文になること**。
        - 並列要素のみを入れ替えても、文全体の意味が大きく変わらないこと（主要な役割・意味構造が維持される）。

        【判定方法】
        - 両方の基準を満たした場合のみ「True」、いずれか一つでも満たさなければ「False」。
        - 出力は必ず以下のJSON形式のみ。判定理由や補足コメントは**絶対に付けない**こと。

        【出力】
        次のJSONのみ（説明は禁止）:
        {{"judge_result": true}} または {{"judge_result": false}}
    """


@dataclass(frozen=True)
class ParallelJudgeConfig:
    # vLLM 側の served_model_name（compose の served_model_name と合わせる）
    model: str = "llmjp-13b"

    # 判定のみなので短い生成に限定
    max_tokens: int = 32
    temperature: float = 0.0

    # キャッシュ（同一 (sentence, elements) の再判定を省略）
    cache_size: int = 4096

    # ログ（研究・再現性向け）
    # - None の場合はログ出力しない
    # - failures_only=True の場合は失敗時のみ出力（推奨）
    log_path: Optional[str] = None
    failures_only: bool = True

    # ログ出力時に巨大テキストを避ける（0 は無制限）
    max_sentence_chars: int = 500
    max_element_chars: int = 200


@dataclass
class ParallelJudgeStats:
    calls: int = 0  # judge_parallel 呼び出し回数
    misses: int = 0  # キャッシュミス回数（= 実際に LLM を呼んだ回数）
    failures: int = 0  # 例外・パース不整合など（結果は False に倒す）
    total_latency_sec: float = 0.0  # judge_parallel 全体（キャッシュヒット含む）
    total_llm_latency_sec: float = 0.0  # LLM 呼び出し（キャッシュミス分のみ）

    @property
    def hits(self) -> int:
        return max(self.calls - self.misses, 0)

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_sec / self.calls * 1000.0) if self.calls else 0.0

    @property
    def avg_llm_latency_ms(self) -> float:
        return (self.total_llm_latency_sec / self.misses * 1000.0) if self.misses else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "failures": self.failures,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_llm_latency_ms": self.avg_llm_latency_ms,
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_elements(xs: List[str]) -> Tuple[str, ...]:
    # キャッシュキー安定化のため、strip と空要素除去のみ行う
    return tuple((x or "").strip() for x in xs if (x or "").strip())


def _clip(s: str, n: int) -> str:
    if n <= 0:
        return s
    return s if len(s) <= n else (s[:n] + "…")


class ParallelJudgeLLMJP:
    def __init__(self, cfg: Optional[ParallelJudgeConfig] = None):
        self.cfg = cfg or ParallelJudgeConfig()

        # HTTP client（Session 再利用のため 1 回だけ生成）
        self.client = get_llmjp_http()

        self.stats = ParallelJudgeStats()

        # LRU cache（内部関数のみキャッシュ）
        self._judge_cached = lru_cache(maxsize=self.cfg.cache_size)(self._judge_uncached)

    def stats_snapshot(self) -> Dict[str, Any]:
        """軽量に統計を取得（CKY 実行後のログ出力などに使用）"""
        return self.stats.to_dict()

    def clear_cache(self) -> None:
        """実験条件を切り替える場合などにキャッシュをクリア"""
        self._judge_cached.cache_clear()

    def judge_parallel(
        self,
        sentence: str,
        parallel_elements: List[str],
        *,
        strict: bool = False,
    ) -> bool:
        """
        仕様: 常に bool を返す（True/False）
        strict=True の場合のみ、内部例外を送出する
        """
        t0 = time.perf_counter()
        self.stats.calls += 1

        s = (sentence or "").strip()
        elems = _normalize_elements(parallel_elements)

        # 入力が不十分なら並列成立とはみなさない
        if (not s) or (len(elems) < 2):
            dt = time.perf_counter() - t0
            self.stats.total_latency_sec += dt
            return False

        misses_before = self.stats.misses

        try:
            out = self._judge_cached(s, elems)
            return bool(out)

        except Exception as e:
            self.stats.failures += 1
            if self.cfg.log_path and (not self.cfg.failures_only):
                self._log_event(
                    kind="exception",
                    sentence=s,
                    elements=list(elems),
                    result=False,
                    cached=(self.stats.misses == misses_before),
                    latency_sec=(time.perf_counter() - t0),
                    error=f"{type(e).__name__}: {e}",
                )
            if strict:
                raise
            return False

        finally:
            dt = time.perf_counter() - t0
            self.stats.total_latency_sec += dt

    def _judge_uncached(self, sentence: str, elements: Tuple[str, ...]) -> bool:
        """キャッシュミス時のみ実行される（= 実際に LLM を呼ぶ）"""
        self.stats.misses += 1
        t0 = time.perf_counter()

        elements_str = ", ".join(f"「{e}」" for e in elements)
        user_prompt = _USER_TMPL.format(sentence=sentence, elements_str=elements_str)

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            obj = self.client.chat_json(
                messages,
                json_schema=_PARALLEL_SCHEMA,
                schema_name="parallel-judge",
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )

            # スキーマ上は boolean のみ想定だが、保険として文字列も解釈
            v = obj.get("judge_result", False)
            if isinstance(v, bool):
                result = v
            elif isinstance(v, str):
                ss = v.strip().lower()
                result = True if ss == "true" else False
            else:
                result = False

            return result

        except Exception as e:
            self.stats.failures += 1
            if self.cfg.log_path:
                self._log_event(
                    kind="failure",
                    sentence=sentence,
                    elements=list(elements),
                    result=False,
                    cached=False,
                    latency_sec=(time.perf_counter() - t0),
                    error=f"{type(e).__name__}: {e}",
                )
            # 仕様: 失敗時は False に倒す（strict は上位で扱う）
            return False

        finally:
            self.stats.total_llm_latency_sec += (time.perf_counter() - t0)

    def _log_event(
        self,
        *,
        kind: str,
        sentence: str,
        elements: List[str],
        result: bool,
        cached: bool,
        latency_sec: float,
        error: Optional[str] = None,
    ) -> None:
        """
        JSONL で 1 行追記。
        デフォルトは failures_only=True を想定（成功を大量に書かない）。
        """
        if not self.cfg.log_path:
            return

        if self.cfg.failures_only and kind not in ("failure", "exception"):
            return

        # 大きすぎるテキストを抑制
        sent = _clip(sentence, self.cfg.max_sentence_chars)
        elems = [_clip(e, self.cfg.max_element_chars) for e in elements]

        event = {
            "ts": _utcnow_iso(),
            "kind": kind,
            "model": self.cfg.model,
            "result": bool(result),
            "cached": bool(cached),
            "latency_ms": round(latency_sec * 1000.0, 3),
            "sentence": sent,
            "elements": elems,
        }
        if error:
            event["error"] = error
        # 直近の統計も添付（軽量）
        event["stats"] = self.stats.to_dict()

        os.makedirs(os.path.dirname(self.cfg.log_path) or ".", exist_ok=True)
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
