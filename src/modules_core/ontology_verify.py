from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm.llmjp_client import get_llmjp_http_for

PID_RE = re.compile(r"^P\d+$")
QID_RE = re.compile(r"^Q\d+$")

_VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "integer"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    prompt_name: str
    prompt_text: str
    arg_names: Tuple[str, ...]


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render_prompt(template: PromptTemplate, values: Dict[str, Any]) -> str:
    formatter = string.Formatter()
    keys = {fname for _, fname, _, _ in formatter.parse(template.prompt_text) if fname}
    safe_values = {k: (values.get(k, "") or "") for k in keys}
    try:
        return template.prompt_text.format_map(_SafeDict(safe_values))
    except Exception:
        out = template.prompt_text
        for k, v in safe_values.items():
            out = out.replace("{" + k + "}", str(v))
        return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def load_prompt_templates(prompts_json_path: str) -> Dict[str, PromptTemplate]:
    data = _load_json(prompts_json_path)
    out: Dict[str, PromptTemplate] = {}
    for p in data.get("prompts", []):
        pid = normalize_text(p.get("id"))
        if not pid:
            continue
        arg_names = tuple(a.get("name", "") for a in p.get("argument_elements", []) if a.get("name"))
        out[pid] = PromptTemplate(
            prompt_id=pid,
            prompt_name=str(p.get("prompt_name", "")),
            prompt_text=str(p.get("prompt_text", "")),
            arg_names=arg_names,
        )
    return out


@lru_cache(maxsize=2)
def load_relation_prompt_map(mapping_json_path: str) -> Dict[str, Any]:
    return _load_json(mapping_json_path)


@lru_cache(maxsize=16)
def load_ontology_labels(ontology_dir: str, ontology_id: str) -> Dict[str, str]:
    ont_id = normalize_text(ontology_id)
    if not ont_id:
        return {}

    candidates: List[str] = []
    if ont_id.startswith("ont_"):
        candidates.append(f"{ont_id[4:]}_ontology_trans_ja.json")
    candidates.append(f"{ont_id}_ontology_trans_ja.json")

    for fname in candidates:
        path = os.path.join(ontology_dir, fname)
        if os.path.exists(path):
            data = _load_json(path)
            labels: Dict[str, str] = {}
            for c in data.get("concepts", []):
                qid = normalize_text(c.get("qid"))
                if not qid:
                    continue
                label = (
                    normalize_text(c.get("label_ja"))
                    or normalize_text(c.get("label_wiki_ja"))
                    or normalize_text(c.get("label"))
                    or qid
                )
                labels[qid] = label
            return labels
    return {}


class RelationPromptResolver:
    def __init__(self, mapping_json_path: str, prompts_json_path: str, ontology_dir: str) -> None:
        self.mapping = load_relation_prompt_map(mapping_json_path)
        self.prompts = load_prompt_templates(prompts_json_path)
        self.ontology_dir = ontology_dir

        rows = self.mapping.get("rows", [])
        self._rows = rows

        by_pid: Dict[str, List[Dict[str, Any]]] = {}
        by_predicate: Dict[str, List[Dict[str, Any]]] = {}
        by_ontology_pid: Dict[Tuple[str, str], Dict[str, Any]] = {}
        by_ontology_predicate: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for row in rows:
            ont = normalize_text(row.get("ontology_id"))
            pid = normalize_text(row.get("pid"))
            pred = normalize_text(row.get("predicate_ja"))
            if pid:
                by_pid.setdefault(pid, []).append(row)
                if ont:
                    by_ontology_pid[(ont, pid)] = row
            if pred:
                by_predicate.setdefault(pred, []).append(row)
                if ont:
                    by_ontology_predicate[(ont, pred)] = row

        self._by_pid = by_pid
        self._by_predicate = by_predicate
        self._by_ontology_pid = by_ontology_pid
        self._by_ontology_predicate = by_ontology_predicate

    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        return self.prompts.get(normalize_text(prompt_id))

    def resolve_relation_row(self, relation: str, ontology_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        rel = normalize_text(relation)
        if not rel:
            return None
        ont = normalize_text(ontology_id) if ontology_id else ""

        if PID_RE.match(rel):
            if ont:
                hit = self._by_ontology_pid.get((ont, rel))
                if hit:
                    return hit
            rows = self._by_pid.get(rel, [])
        else:
            if ont:
                hit = self._by_ontology_predicate.get((ont, rel))
                if hit:
                    return hit
            rows = self._by_predicate.get(rel, [])

        if len(rows) == 1:
            return rows[0]
        return None

    def resolve_concepts(self, row: Dict[str, Any], ontology_id: Optional[str]) -> Tuple[str, str]:
        ont = normalize_text(ontology_id) if ontology_id else normalize_text(row.get("ontology_id"))
        labels = load_ontology_labels(self.ontology_dir, ont) if ont else {}

        domain_concept = normalize_text(row.get("domain_concept_ja"))
        range_concept = normalize_text(row.get("range_concept_ja"))
        domain_qid = normalize_text(row.get("domain_qid"))
        range_qid = normalize_text(row.get("range_qid"))

        domain = domain_concept
        if (not domain) or QID_RE.match(domain):
            qid = domain if QID_RE.match(domain) else domain_qid
            domain = labels.get(qid, qid) if qid else domain

        range_ = range_concept
        if (not range_) or QID_RE.match(range_):
            qid = range_ if QID_RE.match(range_) else range_qid
            range_ = labels.get(qid, qid) if qid else range_

        return domain or "", range_ or ""


@dataclass(frozen=True)
class OntologyJudgeConfig:
    # When None, use the model configured on the underlying HTTP client (LLMJPHTTPClient.model),
    # which is resolved via LLMJP_ONTO_MODEL -> LLMJP_MODEL -> default.
    model: Optional[str] = None
    max_tokens: int = 64
    temperature: float = 0.0
    cache_size: int = 4096


class OntologyJudgeLLMJP:
    def __init__(self, cfg: Optional[OntologyJudgeConfig] = None) -> None:
        self.cfg = cfg or OntologyJudgeConfig()
        # Use ontology-specific vLLM endpoint(s) for better throughput and scaling.
        self.client = get_llmjp_http_for("onto")
        self._judge_cached = lru_cache(maxsize=self.cfg.cache_size)(self._judge_uncached)
        self.last_error: Optional[str] = None
        self.last_meta: Dict[str, Any] = {}
        self._strict = str(os.getenv("ONTOLOGY_VERIFY_STRICT", "1")).strip().lower() not in ("0", "false", "no", "")

    def judge_prompt(self, prompt_text: str) -> int:
        self.last_error = None
        # Clear request metadata so cache hits don't inherit previous request values.
        try:
            setattr(self.client, "last_base_url", None)
            setattr(self.client, "last_url", None)
        except Exception:
            pass
        self.last_meta = {
            "cached": False,
            "model_used": (self.cfg.model or getattr(self.client, "model", None)),
            "base_urls": getattr(self.client, "base_urls", None),
            "base_url": getattr(self.client, "last_base_url", None),
            "request_url": getattr(self.client, "last_url", None),
        }
        if not prompt_text:
            return 0

        # Detect cache hit/miss for logging (best-effort).
        cached = False
        try:
            before = self._judge_cached.cache_info()
        except Exception:
            before = None
        try:
            out = int(self._judge_cached(prompt_text))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.last_meta.update(
                {
                    "cached": False,
                    "base_url": getattr(self.client, "last_base_url", None),
                    "request_url": getattr(self.client, "last_url", None),
                    "error": self.last_error,
                }
            )
            if self._strict:
                mu = self.last_meta.get("model_used")
                bu = self.last_meta.get("base_url")
                ru = self.last_meta.get("request_url")
                raise RuntimeError(
                    f"ontology verify failed (model={mu!r} base_url={bu!r} request_url={ru!r}): {self.last_error}"
                ) from e
            return 0
        finally:
            try:
                after = self._judge_cached.cache_info()
                if before and after and (after.hits > before.hits):
                    cached = True
            except Exception:
                cached = False
            self.last_meta["cached"] = cached
            self.last_meta["base_url"] = getattr(self.client, "last_base_url", None)
            self.last_meta["request_url"] = getattr(self.client, "last_url", None)
        return out

    def _judge_uncached(self, prompt_text: str) -> int:
        messages = [{"role": "user", "content": prompt_text}]
        try:
            obj = self.client.chat_json(
                messages,
                json_schema=_VERDICT_SCHEMA,
                schema_name="ontology-verify",
                # IMPORTANT: Do not override onto server's served model name by default.
                # When cfg.model is None, let the underlying client decide (LLMJP_ONTO_MODEL etc.).
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.last_meta.update(
                {
                    "cached": False,
                    "model_used": (self.cfg.model or getattr(self.client, "model", None)),
                    "base_urls": getattr(self.client, "base_urls", None),
                    "base_url": getattr(self.client, "last_base_url", None),
                    "request_url": getattr(self.client, "last_url", None),
                    "error": self.last_error,
                }
            )
            if self._strict:
                raise
            return 0

        v = obj.get("verdict", 0)
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                try:
                    return int(s)
                except Exception:
                    return 0
        return 0


def prompt_requires_pair(template: PromptTemplate) -> bool:
    return ("arg1" in template.arg_names) and ("arg2" in template.arg_names)


def pick_other_argument(args: List[str], current: str) -> str:
    for a in args:
        if a != current:
            return a
    return "NULL"


@lru_cache(maxsize=2)
def get_ontology_resolver(mapping_json_path: str, prompts_json_path: str, ontology_dir: str) -> RelationPromptResolver:
    return RelationPromptResolver(mapping_json_path, prompts_json_path, ontology_dir)


@lru_cache(maxsize=1)
def get_ontology_judge() -> OntologyJudgeLLMJP:
    return OntologyJudgeLLMJP()
