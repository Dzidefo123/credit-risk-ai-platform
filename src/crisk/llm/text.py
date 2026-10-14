"""Text normalisation shared by retrieval, the mock answerer and the verifiers.

The same normalisation is applied to both sides of every comparison (quote vs source, number vs quote), so
markdown emphasis, line wrapping and dash variants do not cause false failures.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i
if in into is it its itself just me more most my no nor not of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your according much many give tell show please
""".split())

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-", "‘": "'", "’": "'",
                         "“": '"', "”": '"', " ": " "})
_EMPH = re.compile(r"\*\*|__|`")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def norm(text: str) -> str:
    """Normalise for verbatim comparison: dash/quote variants, markdown emphasis, whitespace, case."""
    return _WS.sub(" ", _EMPH.sub("", text.translate(_DASHES))).strip().lower()


def display(text: str) -> str:
    """Readable single-line text without markdown emphasis (case kept)."""
    return _WS.sub(" ", _EMPH.sub("", text.translate(_DASHES))).strip()


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t[-2].isdigit():
        return t[:-1]
    return t


def _raw_tokens(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN.findall(norm(text)):
        out.append(m)
        if any(c in m for c in "._-/"):
            out.extend(p for p in re.split(r"[._\-/]", m) if p)
    return out


def tokens(text: str) -> list[str]:
    """Lower-case tokens; compound tokens (``rb-02``, ``0.294``, ``pd_level``) are kept and also split into parts."""
    return [_stem(t) for t in _raw_tokens(text)]


def content_tokens(text: str) -> list[str]:
    """Tokens without stopwords (checked before stemming) or single letters; digits are always kept."""
    return [_stem(t) for t in _raw_tokens(text) if t.isdigit() or (t not in STOPWORDS and len(t) > 1)]


def term_set(text: str) -> set[str]:
    """Unstemmed tokens including compound parts, for exact term presence checks."""
    return set(_raw_tokens(text))


def numbers(text: str) -> set[str]:
    """Numeric literals as written (thousands separators removed): '83,597' -> '83597', '$2.06M' -> '2.06'."""
    return {m.replace(",", "").rstrip(".") for m in _NUMBER.findall(norm(text))}


SPECIFIC_TERM = re.compile(r"\b(?:[a-z]{1,4}-?\d{2,3}|\d+\.\d+\.\d+|(?:19|20)\d{2})\b")


def specific_terms(text: str) -> set[str]:
    """Terms a supporting passage must contain: codes (F-005, RB-02, R02), versions (1.0.0) and years."""
    return set(SPECIFIC_TERM.findall(norm(text)))
