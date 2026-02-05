from __future__ import annotations

from typing import Any, Iterable, Sequence


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


def _is_particle_token(token: str, upos: str | None, xpos: str | None) -> bool:
    if not token:
        return False
    if upos == "ADP":
        return True
    if xpos:
        if xpos.startswith("助詞") or "助詞" in xpos:
            return True
    return False


def strip_trailing_particles(
    text: str,
    *,
    particles: Iterable[str] | None = None,
    clause: Sequence[Any] | None = None,
) -> str:
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

    # If bunsetsu info is available, remove trailing particles by POS tags.
    # This is more reliable than a static particle list.
    if clause and len(clause) >= 5:
        tokens = list(clause[2] or [])
        upos_list = list(clause[3] or [])
        xpos_list = list(clause[4] or [])
        i = len(tokens) - 1
        while i >= 0 and s:
            token = str(tokens[i]) if i < len(tokens) else ""
            upos = str(upos_list[i]) if i < len(upos_list) else ""
            xpos = str(xpos_list[i]) if i < len(xpos_list) else ""
            if _is_particle_token(token, upos, xpos):
                if len(s) > len(token) and s.endswith(token):
                    s = s[: -len(token)]
                    s = _rstrip_any(s, _TRAILING_PUNCT_CHARS)
                    s = s.rstrip()
                    i -= 1
                    continue
            break
        return s

    # Fallback: Common Japanese particles (strip only trailing; internal "の" stays intact).
    parts = list(particles) if particles is not None else [
        "から",
        "まで",
        "より",
        "が",
        "は",
        "を",
        "に",
        "へ",
        "と",
        "で",
        "も",
        "や",
        "の",
    ]

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
