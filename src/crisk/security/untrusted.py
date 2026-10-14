"""Screening of untrusted text: uploaded documents and copilot questions.

Untrusted text is data, never instructions. This screen flags instruction-like content aimed at an AI system
(prompt injection) so the pipeline can quarantine a document or refuse a question. It is a deterministic first line
of defence, not a guarantee: the stronger controls are structural (no write tools, access filtering before
retrieval, citation and number verification, and a feature gate that only accepts validated extractions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS: dict[str, str] = {
    "ignore_instructions": r"\b(ignore|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?)\b",
    "disregard": r"\bdisregard\b.{0,40}\b(instructions?|rules?|validation|citations?|policy|checks?)\b",
    "role_switch": r"\byou are now\b|\b(administrator|admin|approval|developer|god) mode\b",
    "system_prompt": r"\bsystem\s+(prompt|instructions?|message)\b",
    "role_prefix": r"(^|\n)\s*(system|assistant|developer)\s*:|\[(system|assistant|developer)\]",
    "note_to_ai": r"\bnote to (the )?(ai|llm|model|assistant|reviewer bot)\b|\bai reviewer\b",
    "exfiltration": r"\b(reveal|print|show|list|output)\b.{0,30}\b(api[ _-]?keys?|system prompt|passwords?|secrets?|"
                    r"credentials?)\b",
    "status_override": r"\b(mark|set|flag)\b.{0,40}\b(as )?(approved|verified|validated)\b|\bskip (the )?validation\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in INJECTION_PATTERNS.items()}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


@dataclass(frozen=True)
class ScreenHit:
    pattern: str
    line: int
    excerpt: str


def clean_text(text: str) -> str:
    """Remove control and bidirectional-override characters (keeps newlines and tabs)."""
    return _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def screen_text(text: str) -> list[ScreenHit]:
    hits: list[ScreenHit] = []
    for i, line in enumerate(clean_text(text).split("\n"), start=1):
        for name, rx in _COMPILED.items():
            if rx.search(line):
                hits.append(ScreenHit(name, i, line.strip()[:160]))
    return hits
