from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pattern.pattern_parser import PatternParser
from pattern.pattern_nodes import VariableNode, extract_literal_strings, count_parallel_variables


@dataclass(frozen=True)
class PatternEntry:
    pattern_id: str
    pattern: str
    ast: Any
    var_count: int
    literal_list: List[str]
    parallel_var_count: int
    ast_uid: str


def _count_variables(ast: Any) -> int:
    cnt = 0

    def visit(node: Any) -> None:
        nonlocal cnt
        if isinstance(node, VariableNode):
            cnt += 1
        for attr in ("elements", "options", "block"):
            child = getattr(node, attr, None)
            if not child:
                continue
            if isinstance(child, list):
                for c in child:
                    visit(c)
            else:
                visit(child)

    visit(ast)
    return cnt


def _ast_uid(ast: Any) -> str:
    try:
        b = repr(ast).encode("utf-8", errors="ignore")
    except Exception:
        b = b""
    return md5(b).hexdigest()[:16]


def _load_index_ids(index_path: str, *, statuses: Optional[Set[str]] = None) -> Set[str]:
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids: Set[str] = set()
    for row in data.get("patterns", []):
        pid = str(row.get("pattern_id", "")).strip()
        if not pid:
            continue
        if statuses is not None:
            status = str(row.get("status", "")).strip()
            if status not in statuses:
                continue
        ids.add(pid)
    return ids


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_and_compile_patterns(
    *,
    index_path: str,
    jsonl_path: str,
    statuses: Optional[Sequence[str]] = None,
) -> List[PatternEntry]:
    status_set = set(statuses) if statuses else None
    ids = _load_index_ids(index_path, statuses=status_set)
    if not ids:
        return []

    parser = PatternParser()
    entries: List[PatternEntry] = []

    for row in _iter_jsonl(jsonl_path):
        pid = str(row.get("pattern_id", "")).strip()
        if pid not in ids:
            continue
        pat = str(row.get("pattern", "")).strip()
        if not pat:
            continue
        ast = parser.parse(pat)
        literal_list = extract_literal_strings(ast)
        parallel_var_count = count_parallel_variables(ast)
        var_count = _count_variables(ast)
        entries.append(
            PatternEntry(
                pattern_id=pid,
                pattern=pat,
                ast=ast,
                var_count=var_count,
                literal_list=literal_list,
                parallel_var_count=parallel_var_count,
                ast_uid=_ast_uid(ast),
            )
        )

    return entries


def build_ast_dict(entries: Sequence[PatternEntry]) -> Dict[int, List[Dict[str, Any]]]:
    ast_dict: Dict[int, List[Dict[str, Any]]] = {}
    for e in entries:
        ast_dict.setdefault(e.var_count, []).append(
            {
                "pattern_id": e.pattern_id,
                "pattern": e.pattern,
                "ast": e.ast,
                "var_count": e.var_count,
                "literal_list": e.literal_list,
                "parallel_var_count": e.parallel_var_count,
                "ast_uid": e.ast_uid,
            }
        )
    return ast_dict
