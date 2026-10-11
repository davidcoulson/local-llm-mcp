# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.20,<2", "httpx>=0.27", "numpy>=2"]
# ///
"""
local-llm: an MCP server that lets Claude Code hand bulk or private work to local models.

The models run in Ollama on the Unraid server's RTX 3070 (http://10.2.6.50:11434 by default):
a small chat model (qwen3:8b) for summaries and questions, and an embedding model
(qwen3-embedding:0.6b) for semantic search. File contents go to that server only, never to a
cloud service, and the work does not use Claude's context or allowance.

Configuration (environment):
  OLLAMA_URL        Ollama base URL (default http://10.2.6.50:11434)
  LOCAL_CHAT_MODEL  chat model (default qwen3:8b)
  LOCAL_EMBED_MODEL embedding model (default qwen3-embedding:0.6b)
  LOCAL_LLM_CACHE   cache directory for the search index (default ~/.cache/local-llm-mcp)

Run: uv run --script server.py   (stdio transport)
"""

from __future__ import annotations

import fnmatch
import logging
import os
import sqlite3
import subprocess
from pathlib import Path

import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://10.2.6.50:11434").rstrip("/")
CHAT_MODEL = os.environ.get("LOCAL_CHAT_MODEL", "qwen3:8b")
EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "qwen3-embedding:0.6b")
CACHE_DIR = Path(
    os.environ.get("LOCAL_LLM_CACHE", "~/.cache/local-llm-mcp")
).expanduser()

# The chat model runs with a 16k-token context; keep each request's text well inside it.
CHUNK_CHARS = 24_000
MAX_FILE_BYTES = 64 * 1024 * 1024
# Embedding requests are short, so a small context keeps the embedding model at ~1 GB of VRAM
# and leaves room for the chat model and Plex transcodes on the 8 GB card.
EMBED_CTX = 2048
EMBED_BATCH = 32
SEARCH_LINES = 60
SEARCH_OVERLAP = 10
SEARCH_MAX_BYTES = 512 * 1024
TEXT_SUFFIXES = {
    ".py",
    ".go",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".php",
    ".sh",
    ".bash",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".kt",
    ".swift",
    ".rb",
    ".lua",
    ".sql",
    ".md",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".plg",
    ".page",
    ".html",
    ".css",
    ".scss",
    ".tf",
    ".hcl",
    ".dockerfile",
    ".env.example",
    ".log",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "htmlcov",
    ".tox",
    ".next",
}
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve the most relevant code or text passages\nQuery: "

mcp = FastMCP("local-llm")
# httpx logs every request at INFO; keep stderr for real problems.
logging.getLogger("httpx").setLevel(logging.WARNING)


def _read_text(path: str) -> tuple[Path, str]:
    """Read a text file, refusing directories, missing files and huge files."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"{p} is {size / 1e6:.0f} MB; the limit is {MAX_FILE_BYTES / 1e6:.0f} MB"
        )
    return p, p.read_text(errors="replace")


async def _chat(
    client: httpx.AsyncClient, system: str, user: str, max_tokens: int = 1024
) -> str:
    """
    One non-streaming chat turn with thinking off (faster, and the output is the answer).

    Small models sometimes repeat themselves without end, so output is capped and repetition
    is discouraged; the request also has an overall timeout.
    """
    r = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
                "repeat_penalty": 1.15,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split text into pieces of about `size` characters, at line boundaries where possible."""
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            nl = text.rfind("\n", start + size // 2, end)
            end = nl + 1 if nl > 0 else end
        out.append(text[start:end])
        start = end
    return out


@mcp.tool()
async def local_summarize(path: str, focus: str = "") -> str:
    """
    Summarise a large local text file (logs, syslog, diagnostics, CSV, docs) with the local model.

    Use this instead of reading a big file into context, or for files that should not leave the
    LAN. Long files are summarised in pieces, then combined. `focus` says what to look for
    (e.g. "errors and warnings with timestamps", "anything about disk sdq"). The local model is
    small: treat the result as a lead and verify specific facts in the file.
    """
    p, text = _read_text(path)
    task = focus.strip() or "the main points, errors, warnings and anything unusual"
    system = (
        "You summarise files for an engineer. Be factual and concise, quote exact identifiers, "
        "timestamps and numbers, and never invent details that are not in the text."
    )
    pieces = _chunks(text)
    async with httpx.AsyncClient() as client:
        if len(pieces) == 1:
            return await _chat(
                client, system, f"File: {p.name}\nFocus: {task}\n\n{pieces[0]}"
            )
        notes = []
        for i, piece in enumerate(pieces, 1):
            notes.append(
                await _chat(
                    client,
                    system,
                    f"File: {p.name}, part {i} of {len(pieces)}\nFocus: {task}\n"
                    f"Write short notes on this part only (at most 15 bullet points).\n\n{piece}",
                    max_tokens=400,
                )
            )
        joined = "\n\n".join(f"Part {i}:\n{n}" for i, n in enumerate(notes, 1))
        summary = await _chat(
            client,
            system,
            f"These are notes on the {len(pieces)} parts of {p.name}, in order.\n"
            f"Focus: {task}\nCombine them into one summary. Group repeated events and give "
            f"counts instead of listing each one.\n\n{joined}",
            max_tokens=2048,
        )
    return f"{summary}\n\n({p}: {len(text):,} characters in {len(pieces)} parts, model {CHAT_MODEL})"


@mcp.tool()
async def local_ask(prompt: str, path: str = "") -> str:
    """
    Ask the local model a question, optionally about a local file (first ~24k characters).

    For cheap, private or bulk text work: classifying, extracting fields, rewording, first-pass
    triage. Not for code changes or anything that needs careful reasoning.
    """
    context = ""
    if path:
        p, text = _read_text(path)
        context = f"\n\nFile {p.name}:\n{text[:CHUNK_CHARS]}"
    async with httpx.AsyncClient() as client:
        return await _chat(
            client, "You are a concise, factual assistant.", prompt + context
        )


def _files(root: Path, pattern: str) -> list[Path]:
    """Text files under root: git-tracked ones in a repo, else a filtered walk."""
    try:
        out = (
            subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                capture_output=True,
                check=True,
                timeout=30,
            )
            .stdout.decode()
            .split("\0")
        )
        paths = [root / f for f in out if f]
    except (subprocess.SubprocessError, OSError):
        paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            ]
            paths.extend(Path(dirpath) / f for f in filenames)
    keep = []
    for p in paths:
        if pattern and not fnmatch.fnmatch(str(p.relative_to(root)), pattern):
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name.lower() not in {
            "dockerfile",
            "makefile",
        }:
            continue
        try:
            if p.is_file() and 0 < p.stat().st_size <= SEARCH_MAX_BYTES:
                keep.append(p)
        except OSError:
            continue
    return keep


def _line_chunks(text: str) -> list[tuple[int, int, str]]:
    """(first line, last line, text) windows of SEARCH_LINES lines with some overlap."""
    lines = text.splitlines()
    out = []
    step = SEARCH_LINES - SEARCH_OVERLAP
    for start in range(0, max(len(lines), 1), step):
        part = lines[start : start + SEARCH_LINES]
        if any(s.strip() for s in part):
            out.append((start + 1, start + len(part), "\n".join(part)))
        if start + SEARCH_LINES >= len(lines):
            break
    return out


async def _embed(client: httpx.AsyncClient, texts: list[str]) -> np.ndarray:
    r = await client.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts,
            "truncate": True,
            "keep_alive": "2m",
            "options": {"num_ctx": EMBED_CTX},
        },
        timeout=600,
    )
    r.raise_for_status()
    vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


def _db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_DIR / "index.sqlite")
    db.execute("""CREATE TABLE IF NOT EXISTS chunks (
        model TEXT, path TEXT, sig TEXT, first INTEGER, last INTEGER, text TEXT, vec BLOB)""")
    db.execute("CREATE INDEX IF NOT EXISTS by_path ON chunks (model, path)")
    return db


@mcp.tool()
async def semantic_search(query: str, root: str, k: int = 8, glob: str = "") -> str:
    """
    Search the text and code under a folder by meaning, not by exact words, using local embeddings.

    Good for "where is X handled", "code that does something like Y", or finding passages in docs
    and notes when grep terms are unknown. Indexes git-tracked text files (or a filtered walk
    outside a repo) on first use and only re-embeds changed files later. `glob` narrows the files,
    e.g. "custom_components/**/*.py". Returns file:line ranges with snippets; open the file to
    confirm.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"not a directory: {base}")
    files = _files(base, glob)
    db = _db()
    indexed = embedded = 0
    async with httpx.AsyncClient() as client:
        pending: list[tuple[str, str, int, int, str]] = []
        for f in files:
            st = f.stat()
            sig = f"{st.st_mtime_ns}:{st.st_size}"
            key = str(f)
            row = db.execute(
                "SELECT sig FROM chunks WHERE model=? AND path=? LIMIT 1",
                (EMBED_MODEL, key),
            ).fetchone()
            if row and row[0] == sig:
                indexed += 1
                continue
            db.execute(
                "DELETE FROM chunks WHERE model=? AND path=?", (EMBED_MODEL, key)
            )
            text = f.read_text(errors="replace")
            rel = f.relative_to(base)
            for first, last, chunk in _line_chunks(text):
                pending.append((key, sig, first, last, f"File: {rel}\n{chunk}"))
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i : i + EMBED_BATCH]
            vecs = await _embed(client, [b[4] for b in batch])
            db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                [
                    (EMBED_MODEL, b[0], b[1], b[2], b[3], b[4], v.tobytes())
                    for b, v in zip(batch, vecs)
                ],
            )
            embedded += len(batch)
        db.commit()
        qvec = (await _embed(client, [QUERY_INSTRUCTION + query]))[0]
    keys = [str(f) for f in files]
    if not keys:
        return f"No text files to search under {base}."
    rows = []
    for i in range(0, len(keys), 500):
        part = keys[i : i + 500]
        rows += db.execute(
            f"SELECT path, first, last, text, vec FROM chunks WHERE model=? AND path IN ({','.join('?' * len(part))})",
            (EMBED_MODEL, *part),
        ).fetchall()
    db.close()
    mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(
        len(rows), -1
    )
    scores = mat @ qvec
    best = np.argsort(-scores)[: max(1, min(k, 30))]
    out = []
    for j in best:
        path, first, last, text, _ = rows[j]
        snippet = "\n".join(text.splitlines()[1:13])
        out.append(
            f"{Path(path).relative_to(base)}:{first}-{last}  (score {scores[j]:.3f})\n{snippet}"
        )
    head = (
        f"{len(files)} files under {base} ({embedded} chunks embedded now, "
        f"{indexed} files unchanged since the last index)\n"
    )
    return head + "\n\n".join(out)


if __name__ == "__main__":
    mcp.run()
