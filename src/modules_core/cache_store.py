from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, Optional


@dataclass
class SentenceCacheStore:
    root_dir: str
    suffix: str

    def __post_init__(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)

    def _key(self, sentence: str) -> str:
        h = sha1(sentence.encode("utf-8", errors="ignore")).hexdigest()
        return f"{h}.{self.suffix}.json.gz"

    def _path(self, sentence: str) -> str:
        return os.path.join(self.root_dir, self._key(sentence))

    def load(self, sentence: str) -> Optional[Dict[str, Any]]:
        path = self._path(sentence)
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def save(self, sentence: str, payload: Dict[str, Any]) -> None:
        path = self._path(sentence)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_many(self, sentences: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for s in sentences:
            data = self.load(s)
            if data is not None:
                out[s] = data
        return out
