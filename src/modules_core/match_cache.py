from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional

from modules_core.cache_store import SentenceCacheStore


# v2: varmap_clean/X_values/Y_values now preserve internal spaces (important for Wikidata lookup),
# and particles are stripped only at the end (not removed from the middle).
SCHEMA_VERSION = 2


def compute_patterns_fingerprint(ast_dict: Dict[int, List[Dict[str, Any]]]) -> str:
    uids: List[str] = []
    for _v, entries in (ast_dict or {}).items():
        for e in entries or []:
            uid = str((e or {}).get("ast_uid", "")).strip()
            if uid:
                uids.append(uid)
    uids.sort()
    payload = "|".join(uids)
    return sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _is_list_str(xs: Any) -> bool:
    return isinstance(xs, list) and all(isinstance(x, str) for x in xs)


def validate_match_cache_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        return False
    if not isinstance(payload.get("patterns_fingerprint", None), str):
        return False
    if not isinstance(payload.get("sentence", None), str):
        return False
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return False

    for m in matches:
        if not isinstance(m, dict):
            return False
        for k in (
            "ast_uid",
            "pattern_id",
            "pattern",
            "var_count",
            "parallel_var_count",
            "literal_list",
            "parallel_var_groups",
            "varmap_raw",
            "varmap_clean",
            "X_values",
            "Y_values",
        ):
            if k not in m:
                return False
        if not isinstance(m.get("ast_uid"), str):
            return False
        if not isinstance(m.get("pattern_id"), str):
            return False
        if not isinstance(m.get("pattern"), str):
            return False
        if not isinstance(m.get("var_count"), int):
            return False
        if not isinstance(m.get("parallel_var_count"), int):
            return False
        if not _is_list_str(m.get("literal_list")):
            return False
        if not (isinstance(m.get("parallel_var_groups"), list) and all(_is_list_str(g) for g in m.get("parallel_var_groups"))):
            return False
        if not isinstance(m.get("varmap_raw"), dict):
            return False
        if not isinstance(m.get("varmap_clean"), dict):
            return False
        if not _is_list_str(m.get("X_values")):
            return False
        if not _is_list_str(m.get("Y_values")):
            return False

    return True


@dataclass
class MatchCacheStore:
    store: SentenceCacheStore
    patterns_fingerprint: str

    @classmethod
    def from_dir(cls, root_dir: str, patterns_fingerprint: str) -> "MatchCacheStore":
        return cls(SentenceCacheStore(root_dir, "match"), patterns_fingerprint)

    def load(self, sentence: str) -> Optional[Dict[str, Any]]:
        data = self.store.load(sentence)
        if not data:
            return None
        if not validate_match_cache_payload(data):
            return None
        if str(data.get("patterns_fingerprint", "")) != str(self.patterns_fingerprint):
            return None
        return data

    def save(self, sentence: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "patterns_fingerprint": self.patterns_fingerprint,
            "sentence": sentence,
            "matches": matches,
        }
        self.store.save(sentence, payload)
        return payload

    def load_many(self, sentences: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for s in sentences:
            data = self.load(s)
            if data is not None:
                out[s] = data
        return out
