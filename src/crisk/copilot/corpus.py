"""Approved copilot corpus: versioned sources split into passages, with role-based access applied before retrieval.

A source's version is the SHA-256 of its content (LF-normalised); its as-of date is its last commit date (``git
log``), or ``uncommitted`` when the working copy differs from the last commit. Passages keep their line range, so a
citation can be located in the exact source version.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from crisk.llm.text import content_tokens

_SEP = re.compile(r"^\|[\s:\-|]+\|$")

DEFAULT_MANIFEST = Path("configs/copilot/corpus_v1.yaml")


@dataclass(frozen=True)
class Passage:
    pid: str
    source_id: str
    path: str
    version: str
    as_of: str
    roles: frozenset[str]
    line_start: int
    line_end: int
    section: str
    text: str


def _git_as_of(root: Path, path: str) -> str:
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", path], cwd=root, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        date = subprocess.run(["git", "log", "-1", "--format=%cs", "--", path], cwd=root, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return "uncommitted" if dirty or not date else date


def _chunk_markdown(lines: list[str]) -> list[tuple[int, int, str, list[str]]]:
    """Paragraph blocks (a heading is attached to the block that follows it) and one passage per table row.
    A table-row passage holds the header row and the data row (both verbatim lines), so it can be cited alone."""
    out: list[tuple[int, int, str, list[str]]] = []
    block: list[str] = []
    table: list[tuple[int, str]] = []
    start, section, heading = 0, "", None

    def flush_block(end: int) -> None:
        nonlocal block
        if block:
            out.append((start, end, section, block))
            block = []

    def flush_table() -> None:
        nonlocal table, heading
        if not table:
            return
        if heading:
            out.append((heading[0], heading[0], section, [heading[1]]))
            heading = None
        (h_no, header), rows = table[0], [(n, r) for n, r in table[1:] if not _SEP.match(r.strip())]
        for n, r in rows:
            out.append((h_no, n, section, [header, r]))
        table = []

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("|"):
            flush_block(i - 1)
            table.append((i, line))
            continue
        flush_table()
        if line.startswith("#"):
            flush_block(i - 1)
            if heading:
                out.append((heading[0], heading[0], section, [heading[1]]))
            section = line.lstrip("#").strip()
            heading = (i, line)
            continue
        if not line.strip():
            flush_block(i - 1)
            continue
        if not block:
            start = i
            if heading:
                start, block = heading[0], [heading[1]]
                heading = None
        block.append(line)
    flush_table()
    flush_block(len(lines))
    if heading:
        out.append((heading[0], heading[0], section, [heading[1]]))
    return out


def _chunk_yaml(lines: list[str]) -> list[tuple[int, int, str, list[str]]]:
    """Comment blocks, top-level keys and list items (with their continuation lines) become passages."""
    out, block, start, kind = [], [], 0, None

    def flush(end: int) -> None:
        nonlocal block
        if block:
            out.append((start, end, "", block))
            block = []

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            flush(i - 1)
            kind = None
            continue
        this = "comment" if s.startswith("#") else ("item" if s.startswith("- ") else
                                                    ("top" if not line[0].isspace() else "cont"))
        new = this in ("item", "top") or (this == "comment" and kind != "comment")
        if new or not block:
            flush(i - 1)
            start = i
        block.append(line)
        kind = this if this != "cont" else kind
    flush(len(lines))
    return out


class Corpus:
    def __init__(self, manifest: dict, root: Path):
        self.corpus_id = manifest["corpus_id"]
        self.root = Path(root)
        groups = {"all": set(manifest["all_roles"]), "model_risk": set(manifest["model_risk_roles"])}
        self.sources: dict[str, dict] = {}
        self.passages: list[Passage] = []
        for src in manifest["sources"]:
            raw = (self.root / src["path"]).read_bytes().replace(b"\r\n", b"\n")
            text = raw.decode("utf-8")
            roles = frozenset(groups[src["roles"]] if isinstance(src["roles"], str) else src["roles"])
            version = hashlib.sha256(raw).hexdigest()
            meta = {"source_id": src["id"], "path": src["path"], "version": version, "as_of": _git_as_of(
                self.root, src["path"]), "roles": sorted(roles), "text": text}
            self.sources[src["id"]] = meta
            lines = text.split("\n")
            title = next((ln.lstrip("#").strip() for ln in lines if ln.startswith("# ")), src["path"])
            chunks = _chunk_yaml(lines) if src["path"].endswith((".yaml", ".yml")) else _chunk_markdown(lines)
            for n, (a, b, section, block) in enumerate(chunks):
                self.passages.append(Passage(f"{src['id']}#{n}", src["id"], src["path"], version, meta["as_of"],
                                             roles, a, b, f"{title} / {section}" if section else title,
                                             "\n".join(block)))

    @classmethod
    def load(cls, manifest_path: Path | str = DEFAULT_MANIFEST, root: Path | str = ".") -> Corpus:
        root = Path(root)
        p = Path(manifest_path)
        manifest = yaml.safe_load((p if p.is_absolute() else root / p).read_text(encoding="utf-8"))
        return cls(manifest, root)

    def for_role(self, role: str) -> list[Passage]:
        return [p for p in self.passages if role in p.roles]

    def readable(self, role: str) -> set[str]:
        return {sid for sid, s in self.sources.items() if role in s["roles"]}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len, strict=True):
            s = 0.0
            for t in set(q):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Retriever:
    """BM25 over the passages a role may read. Built per role, so restricted passages are never scored."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.bm25 = BM25([content_tokens(p.section + " " + p.text) for p in passages])

    def search(self, question: str, k: int = 5) -> list[tuple[Passage, float]]:
        s = self.bm25.scores(content_tokens(question))
        ranked = sorted(range(len(s)), key=lambda i: (-s[i], i))
        return [(self.passages[i], round(s[i], 4)) for i in ranked[:k] if s[i] > 0]
