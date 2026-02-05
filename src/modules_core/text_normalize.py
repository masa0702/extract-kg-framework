from __future__ import annotations

from typing import Iterable


_TRAILING_PUNCT_CHARS = "".join(
    [
        " ",
        "\t",
        "\n",
        "\r",
        "　",  # full-width space
        "、",
        "。",
        "，",
        "．",
        ",",
        ".",
        "!",
        "?",
        "！",
        "？",
        "…",
        "：",
        ":",
        "；",
        ";",
        "）",
        ")",
        "】",
        "]",
        "」",
        "』",
        "”",
        "“",
        "\"",
        "'",
    ]
)


def _rstrip_any(s: str, chars: str) -> str:
    return s.rstrip(chars)


def strip_trailing_particles(text: str, *, particles: Iterable[str] | None = None) -> str:
    """
    Extracted entity/argument normalization.

    - Keep internal particles (e.g., "太郎の車") untouched.
    - Strip *only trailing* particles (e.g., "映画が" -> "映画", "太郎の車は" -> "太郎の車").
    - Preserve internal spaces (important for Wikidata search like "New York").
    """
    if text is None:
        return ""
    s = str(text)

    # Strip only edges first; keep internal spaces.
    s = _rstrip_any(s.lstrip(), "\t\n\r　 ")
    if not s:
        return ""

    # Remove trailing punctuation/symbols first (e.g., "映画が、" -> "映画が").
    s = _rstrip_any(s, _TRAILING_PUNCT_CHARS)
    if not s:
        return ""

    # Common Japanese particles (strip only trailing; internal "の" stays intact).
    parts = list(particles) if particles is not None else ["から", "まで", "より", "が", "は", "を", "に", "へ", "と", "で", "も", "や", "の"]

    changed = True
    while changed and s:
        changed = False
        for p in parts:
            if not p:
                continue
            if len(s) <= len(p):
                continue
            if s.endswith(p):
                s = s[: -len(p)]
                s = _rstrip_any(s, _TRAILING_PUNCT_CHARS)
                changed = True
                break

    return s
