"""Provider-neutral LLM interface.

The pipeline only talks to two task-level methods, so providers are interchangeable and every output passes the
same deterministic verifiers:

- ``answer(question, evidence)`` returns an :class:`AnswerDraft`: an answer with verbatim quotes from evidence ids,
  or an abstention.
- ``extract(doc_type, pages, fields)`` returns ``{field: {"value", "page", "quote"}}`` for one document.

Providers: ``mock`` (default; deterministic, no network) and ``anthropic`` (optional; needs the ``llm`` dependency
group and an API key in the environment, never in the repository). Select with ``configs/llm/llm_v1.yaml`` or the
``CRISK_LLM_PROVIDER`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml


class LLMError(RuntimeError):
    """The provider returned something unusable (bad JSON, truncated output)."""


class LLMUnavailable(LLMError):
    """The provider could not be reached (network, rate limit, server error, missing credentials)."""


class LLMRefusal(LLMError):
    """The provider declined the request."""


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source_id: str
    title: str = ""
    as_of: str | None = None


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    quote: str


@dataclass
class AnswerDraft:
    abstain: bool
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    reason: str | None = None


@dataclass
class Usage:
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LLM(Protocol):
    name: str
    usage: Usage

    def answer(self, question: str, evidence: list[Evidence]) -> AnswerDraft: ...

    def extract(self, doc_type: str, pages: list[str], fields: dict[str, str]) -> dict[str, dict]: ...


DEFAULT_CONFIG = Path("configs/llm/llm_v1.yaml")


def get_llm(config_path: str | Path | None = None, provider: str | None = None) -> LLM:
    path = Path(config_path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    provider = provider or os.environ.get("CRISK_LLM_PROVIDER") or cfg.get("provider", "mock")
    if provider == "mock":
        from crisk.llm.mock import MockLLM

        return MockLLM(**(cfg.get("mock") or {}))
    if provider == "anthropic":
        from crisk.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(**(cfg.get("anthropic") or {}))
    raise ValueError(f"unknown LLM provider {provider!r} (mock | anthropic)")
