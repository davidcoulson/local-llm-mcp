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
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
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
    is discouraged; the request also has an overall timeout. Models without a thinking mode
    reject the `think` field, so the request is retried without it.
    """
    body = {
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
    }
    r = await client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
    if r.status_code == 400 and "think" in r.text.lower():
        del body["think"]
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
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


# --- Log mode -----------------------------------------------------------------------------
#
# A small model summarising a big log in chunks loses one-off events, miscounts repeats and
# spends its words on noise. So for logs a deterministic pre-pass turns the file into a digest
# of a few KB: every message is reduced to a template (numbers, IDs and addresses masked), and
# each template is listed once with its exact count, first and last time and one real example.
# Errors, warnings and hardware events are always listed; routine chatter is reduced to counts.
# Only the digest goes to the model.

DIGEST_FLAGGED_CHARS = 8_000
DIGEST_RARE_CHARS = 3_000
DIGEST_FOCUS_CHARS = 3_000
DIGEST_ROUTINE_CHARS = 2_500
DIGEST_EXAMPLE_CHARS = 220
RARE_MAX = 2  # templates seen at most this often count as rare
FEW_MAX = 3  # flagged templates seen at most this often are listed as one-off problems
PROBLEMS_CHARS = 3_000  # exact problem list returned with the summary
RARE_PER_PROGRAM = 5
NEAR_S = 5  # rare lines this close to a one-off problem are shown with it
NEAR_BEFORE, NEAR_AFTER = 2, 5
BURST_GAP_S = 300  # a quiet gap this long separates runs of a periodic job
_TIMES_KEPT = 500

_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_TS_BSD = re.compile(
    rf"^(?P<ts>(?:{_MONTHS}) [ \d]\d \d\d:\d\d:\d\d)(?:\.\d+)?\s+(?P<rest>.*)$"
)
_TS_ISO = re.compile(
    r"^\[?(?P<ts>\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d)(?:[.,]\d+)?(?:Z|[+-]\d\d:?\d\d)?\]?\s+(?P<rest>.*)$"
)
# "host prog[pid]: message"; the program may contain spaces ("SAS Assist v2024.11.25").
_HOST_PROG = re.compile(
    r"^(?P<host>[\w.-]+)\s+(?P<prog>[^\s\[:][^\[:]{0,40}?)(?:\[\d+\])?:\s(?P<msg>.*)$"
)
_LEVEL = re.compile(
    r"^(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL)\b", re.IGNORECASE
)
_KERNEL_TS = re.compile(r"^\[\s*\d+\.\d+\]\s*")

# Applied in order; earlier patterns protect their matches from later ones.
_MASKS = [
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "<UUID>",
    ),
    (
        re.compile(r"\b\d{4}[-/]\d\d[-/]\d\d(?:[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?)?\b"),
        "<DATE>",
    ),
    (re.compile(r"\b\d{1,2}:\d\d:\d\d(?:[.,]\d+)?\b"), "<TIME>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b", re.IGNORECASE), "<MAC>"),
    (
        re.compile(
            r"(?<![\w:])(?=[0-9a-f:]*::|(?:[0-9a-f]{1,4}:){3})[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}(?:%\w+)?",
            re.IGNORECASE,
        ),
        "<IP6>",
    ),
    (re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE), "<HEX>"),
    (re.compile(r"\bveth[0-9a-f]{5,}\b"), "veth<ID>"),
    (
        re.compile(
            r"(?=[A-Za-z0-9+]*\d)(?=[A-Za-z0-9+]*[a-z])(?=[A-Za-z0-9+]*[A-Z])\b[A-Za-z0-9+]{20,}={0,2}"
        ),
        "<KEY>",
    ),
    (
        re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b", re.IGNORECASE),
        "<HEX>",
    ),
    # Long mixed tokens with several digits: message IDs, serials, session tokens.
    (
        re.compile(
            r"\b(?=[a-z0-9]*\d[a-z0-9]*\d[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{12,}(?:\.\d+)?\b",
            re.IGNORECASE,
        ),
        "<ID>",
    ),
    # Numbers, with an optional short unit (12GiB, 60%, 382.375s); not digits inside words.
    (
        re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?:[a-z%]{1,4})?(?![\w.])", re.IGNORECASE),
        "<N>",
    ),
]

_FLAGS = [
    ("crit", re.compile(r"\b(crit(ical)?|emerg(ency)?|panic|fatal)\b", re.IGNORECASE)),
    ("error", re.compile(r"\b(errors?|err|exception|traceback)\b", re.IGNORECASE)),
    ("fail", re.compile(r"\bfail(ed|ure|ures|ing|s)?\b", re.IGNORECASE)),
    (
        "hw",
        re.compile(
            r"usb disconnect|\bata\d+(\.\d+)?:|\bnvme\d*(n\d+)?(p\d+)?:|\bnvme nvme\d|\bmce\b|machine check|hardware error|i/o error"
            r"|link (is )?down|segfault|\boom\b|out of memory|oom-kill|call trace|medium error"
            r"|blk_update_request|\baer\b|corrected error|thermal|throttl|under-?voltage|tainted",
            re.IGNORECASE,
        ),
    ),
    (
        "warn",
        re.compile(
            r"\bwarn(ing)?s?\b|could not|couldn't|cannot|can't|unable to|does not exist"
            r"|not found|mismatch|denied|timed? ?out|refused|\bstale\b",
            re.IGNORECASE,
        ),
    ),
]
# UPS daemons report power events in ordinary words ("charging, but ..."): treat all as hardware.
_HW_PROGRAMS = re.compile(r"ups|apcupsd|\bnut\b", re.IGNORECASE)

# Routine chatter, reduced to one line each unless the focus mentions one of the keywords.
# Flagged lines (e.g. sshd "Failed password") are never treated as noise.
_NOISE = [
    (
        "sshd logins and sessions",
        re.compile(r"^sshd"),
        None,
        ("ssh", "login", "session", "auth"),
    ),
    (
        "cron jobs",
        re.compile(r"^(cron|crond|anacron|run-parts)$", re.IGNORECASE),
        None,
        ("cron",),
    ),
    (
        "disk spin-down/spin-up and SMART reads",
        None,
        re.compile(
            r"spin(ning)?[ -]?(down|up)|spindown|read SMART|standby", re.IGNORECASE
        ),
        ("spin", "standby", "smart", "sleep"),
    ),
    (
        "docker network interfaces (veth/bridge)",
        None,
        re.compile(r"veth|\bdocker\d*: port|\bbr-[0-9a-f]+: port", re.IGNORECASE),
        ("docker", "veth", "container", "bridge"),
    ),
]
_STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "any",
    "all",
    "about",
    "anything",
    "what",
    "show",
    "list",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "timestamps",
    "times",
    "time",
    "events",
    "unusual",
    "main",
    "points",
    "log",
    "logs",
    "lines",
    "line",
    "please",
    "summary",
    "errors",
    "error",
    "warnings",
    "warning",
    "issues",
    "problems",
    "happened",
    "file",
}


@dataclass
class LogEntry:
    when: datetime | None
    stamp: str
    program: str
    message: str


@dataclass
class Template:
    program: str
    key: str
    example: str
    first: datetime | None
    count: int = 0
    last: datetime | None = None
    times: list[datetime] = field(default_factory=list)
    flags: tuple[str, ...] = ()
    noise: str = ""
    seq: int = 0  # line number of the first occurrence
    near: list[Template] = field(default_factory=list)


@dataclass
class Digest:
    text: str
    lines: int
    templates: int
    problems: str = (
        ""  # flagged messages, one line each, for returning with the summary
    )


def _parse_when(stamp: str) -> datetime | None:
    for fmt in ("%Y %b %d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # BSD syslog has no year; a leap year accepts Feb 29.
            return datetime.strptime(
                f"2000 {stamp}" if fmt[0:3] == "%Y " else stamp, fmt
            )
        except ValueError:
            continue
    return None


def _split_line(line: str) -> tuple[str, str] | None:
    """(timestamp, rest) if the line starts with a syslog or ISO timestamp."""
    m = _TS_BSD.match(line) or _TS_ISO.match(line)
    return (m["ts"], m["rest"]) if m else None


def looks_like_log(text: str, sample: int = 200) -> bool:
    """True if most of the first non-empty lines start with a timestamp."""
    lines = [ln for ln in text.splitlines()[: sample * 2] if ln.strip()][:sample]
    if len(lines) < 3:
        return False
    return sum(1 for ln in lines if _split_line(ln)) >= 0.6 * len(lines)


def parse_log_line(line: str, prev: LogEntry | None = None) -> LogEntry:
    """Split a log line into time, program and message (host and PID dropped)."""
    split = _split_line(line)
    if not split:
        # A continuation line (stack trace, wrapped message) belongs to the previous entry.
        return LogEntry(
            prev.when if prev else None,
            prev.stamp if prev else "",
            prev.program if prev else "",
            line.strip(),
        )
    stamp, rest = split
    program, message = "", rest
    m = None if _LEVEL.match(rest) else _HOST_PROG.match(rest)
    if m:
        program, message = m["prog"].strip(), m["msg"]
    return LogEntry(
        _parse_when(stamp), stamp, program, _KERNEL_TS.sub("", message).strip()
    )


def normalize_message(message: str) -> str:
    """Mask the variable parts of a message so repeats of the same event share one template."""
    for pattern, repl in _MASKS:
        message = pattern.sub(repl, message)
    return re.sub(r"\s+", " ", message).strip()


def classify(program: str, message: str) -> tuple[str, ...]:
    """Flag categories (crit, error, fail, hw, warn) that a message falls into."""
    flags = tuple(name for name, pattern in _FLAGS if pattern.search(message))
    if _HW_PROGRAMS.search(program) and "hw" not in flags:
        flags += ("hw",)
    return flags


def _noise_category(program: str, message: str, focus: str) -> str:
    for name, prog_re, msg_re, keywords in _NOISE:
        if any(k in focus for k in keywords):
            continue
        if (prog_re and prog_re.search(program)) or (msg_re and msg_re.search(message)):
            return name
    return ""


def _focus_words(focus: str) -> list[str]:
    words = []
    for w in re.findall(r"[\w./-]{3,}", focus.lower()):
        if w in _STOPWORDS:
            continue
        words.append(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return words


def _focus_pattern(words: list[str]) -> re.Pattern[str] | None:
    """Match focus words at the start of a word: "ups" finds "usbhid-ups", not "backups"."""
    if not words:
        return None
    return re.compile("|".join(rf"\b{re.escape(w)}" for w in words), re.IGNORECASE)


def _period(times: list[datetime]) -> float | None:
    """Seconds between occurrences if they recur at a steady interval (within 10%)."""
    if len(times) < 3:
        return None
    gaps = sorted((b - a).total_seconds() for a, b in pairwise(times))
    mid = gaps[len(gaps) // 2]
    if mid < 60 or any(abs(g - mid) > 0.1 * mid for g in gaps):
        return None
    return mid


def _every(seconds: float) -> str:
    if seconds >= 5400:
        return f"every ~{seconds / 3600:.0f} h"
    return f"every ~{seconds / 60:.0f} min"


def build_digest(text: str, focus: str = "", name: str = "log") -> Digest:
    """Reduce a log to a compact digest of message templates with exact counts and times."""
    focus_l = focus.lower()
    focus_words = _focus_words(focus)
    templates: dict[tuple[str, str], Template] = {}
    programs: dict[str, int] = {}
    bursts: dict[str, list[datetime]] = {}
    last_seen: dict[str, datetime] = {}
    first = last = None
    nlines = 0
    prev = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        nlines += 1
        entry = parse_log_line(raw, prev)
        prev = entry
        when = entry.when
        if when:
            first = first or when
            last = when
        programs[entry.program] = programs.get(entry.program, 0) + 1
        if when and entry.program:
            seen = last_seen.get(entry.program)
            if seen is None or (when - seen).total_seconds() > BURST_GAP_S:
                bursts.setdefault(entry.program, []).append(when)
            last_seen[entry.program] = when
        key = (entry.program, normalize_message(entry.message))
        t = templates.get(key)
        if t is None:
            flags = classify(entry.program, entry.message)
            t = templates[key] = Template(
                program=entry.program,
                key=key[1],
                example=entry.message,
                first=when,
                flags=flags,
                seq=nlines,
                noise=""
                if flags
                else _noise_category(entry.program, entry.message, focus_l),
            )
        t.count += 1
        t.last = when
        if when and len(t.times) < _TIMES_KEPT:
            t.times.append(when)

    one_day = bool(first and last and first.date() == last.date())

    def ts(d: datetime | None) -> str:
        if d is None:
            return "?"
        return d.strftime("%H:%M:%S" if one_day else "%b %d %H:%M:%S")

    def span(t: Template) -> str:
        if t.count == 1 or t.first == t.last:
            return ts(t.first)
        out = f"{ts(t.first)}-{ts(t.last)}"
        period = _period(t.times) if len(t.times) == t.count else None
        return f"{out}, {_every(period)}" if period else out

    def clip(s: str, n: int = DIGEST_EXAMPLE_CHARS) -> str:
        return s if len(s) <= n else s[: n - 3] + "..."

    def entry_line(t: Template) -> str:
        tag = f"[{','.join(t.flags)}] " if t.flags else ""
        prog = f"{t.program}: " if t.program else ""
        out = f"{t.count}x {span(t)} {tag}{prog}{clip(t.example)}"
        for n in t.near:
            out += f"\n    near: {ts(n.first)} {n.program}: {clip(n.example, 160)}"
        return out

    def take(items: list[Template], budget: int, order) -> tuple[list[Template], int]:
        """Pick items (in the given priority order) until the budget is spent."""
        chosen, used = [], 0
        for t in items:
            size = len(entry_line(t)) + 1
            if used + size > budget:
                continue
            chosen.append(t)
            used += size
        return sorted(chosen, key=order), len(items) - len(chosen)

    by_time = lambda t: (t.first or datetime.min, t.program, t.key)
    all_t = list(templates.values())
    flagged = [t for t in all_t if t.flags]
    rest = [t for t in all_t if not t.flags and not t.noise]
    noise = [t for t in all_t if t.noise]

    # Give each one-off problem the rare lines logged just around it (a device coming back,
    # the check that passed a moment later), so the model can tell the whole event.
    pool = sorted(
        (t for t in rest if t.count <= RARE_MAX and t.first), key=lambda t: t.seq
    )
    used: set[int] = set()
    for t in sorted(flagged, key=lambda t: t.seq):
        if t.count > FEW_MAX or not t.first:
            continue
        close = [
            r
            for r in pool
            if id(r) not in used and abs((r.first - t.first).total_seconds()) <= NEAR_S
        ]
        after = [r for r in close if r.seq > t.seq][:NEAR_AFTER]
        before = [r for r in close if r.seq < t.seq][-NEAR_BEFORE:]
        t.near = before + after
        used.update(id(r) for r in t.near)
    rest = [t for t in rest if id(t) not in used]

    sections: list[str] = []
    when_line = f"{ts(first)} to {ts(last)}" if first else "no timestamps found"
    if one_day and first:
        when_line = (
            f"{first.strftime('%b %d')}, {when_line} (times below are on that day)"
        )
    top_progs = sorted(programs.items(), key=lambda kv: -kv[1])
    sections.append(
        f"LOG DIGEST of {name}: {nlines:,} lines, {len(templates):,} distinct messages, {when_line}.\n"
        "Lines by program: "
        + ", ".join(f"{p or '(none)'} {n:,}" for p, n in top_progs[:15])
        + (f", {len(top_progs) - 15} more programs" if len(top_progs) > 15 else "")
        + "\nFormat: COUNTx FIRST-LAST [flags] program: one real example. Counts are exact; "
        "numbers and IDs differ between repeats."
    )

    # Flagged: rare ones and the serious categories first, then the most frequent.
    severity = {"crit": 0, "hw": 1, "error": 2, "fail": 3, "warn": 4}
    prio = sorted(
        flagged,
        key=lambda t: (t.count > FEW_MAX, min(severity[f] for f in t.flags), -t.count),
    )
    chosen, dropped = take(prio, DIGEST_FLAGGED_CHARS, by_time)
    once = [t for t in chosen if t.count <= FEW_MAX]
    repeated = [t for t in chosen if t.count > FEW_MAX]
    sections.append(
        f"== Problems seen only 1-{FEW_MAX} times ({len(once)}) ==\n"
        + ("\n".join(entry_line(t) for t in once) or "(none)")
    )
    body = "\n".join(entry_line(t) for t in repeated) or "(none)"
    if dropped:
        body += f"\n({dropped} more flagged messages omitted for space)"
    sections.append(
        f"== Repeating errors, warnings and hardware events ({len(repeated)}) ==\n{body}"
    )

    focus_re = _focus_pattern(focus_words)
    if focus_re:
        matches = [
            t for t in rest + noise if focus_re.search(f"{t.program} {t.example}")
        ]
        matches.sort(key=lambda t: -t.count)
        chosen, dropped = take(matches, DIGEST_FOCUS_CHARS, by_time)
        if chosen:
            body = "\n".join(entry_line(t) for t in chosen)
            if dropped:
                body += f"\n({dropped} more matching messages omitted)"
            sections.append(
                f"== Other lines matching the focus ({', '.join(focus_words)}) ==\n{body}"
            )
        shown = {id(t) for t in chosen}
        rest = [t for t in rest if id(t) not in shown]

    rare = sorted((t for t in rest if t.count <= RARE_MAX), key=by_time)
    per_prog: dict[str, int] = {}
    capped, extra = [], {}
    for t in rare:
        per_prog[t.program] = per_prog.get(t.program, 0) + 1
        if per_prog[t.program] <= RARE_PER_PROGRAM:
            capped.append(t)
        else:
            extra[t.program] = extra.get(t.program, 0) + 1
    # One-offs first, then lines from the quietest programs (an unusual source is news).
    capped.sort(key=lambda t: (t.count, programs[t.program], by_time(t)))
    chosen, dropped = take(capped, DIGEST_RARE_CHARS, by_time)
    if chosen:
        body = "\n".join(entry_line(t) for t in chosen)
        notes = [f"{n} more from {p or '(none)'}" for p, n in extra.items()]
        if dropped:
            notes.append(f"{dropped} more omitted for space")
        if notes:
            body += f"\n({'; '.join(notes)})"
        sections.append(f"== Other rare lines (seen 1-{RARE_MAX} times) ==\n{body}")

    # Routine: per program, how much and how often, with its commonest messages.
    routine: dict[str, list[Template]] = {}
    for t in rest:
        if t.count > RARE_MAX:
            routine.setdefault(t.program, []).append(t)
    lines, used = [], 0
    for prog, items in sorted(
        routine.items(), key=lambda kv: -sum(t.count for t in kv[1])
    ):
        items.sort(key=lambda t: -t.count)
        n = sum(t.count for t in items)
        runs = bursts.get(prog, [])
        period = _period(runs)
        how = f", in {len(runs)} runs {_every(period)}" if period else ""
        common = "; ".join(f"{t.count}x '{clip(t.key, 90)}'" for t in items[:2])
        line = f"{prog or '(none)'}: {n:,} lines in {len(items)} repeating messages{how}; commonest: {common}"
        if used + len(line) > DIGEST_ROUTINE_CHARS:
            lines.append(
                f"({len(routine) - len(lines)} more programs with routine messages)"
            )
            break
        lines.append(line)
        used += len(line) + 1
    if lines:
        sections.append(
            "== Routine repeating messages (not flagged) ==\n" + "\n".join(lines)
        )

    groups: dict[str, list[Template]] = {}
    for t in noise:
        groups.setdefault(t.noise, []).append(t)
    if groups:
        lines = []
        for cat, items in groups.items():
            items.sort(key=lambda t: -t.count)
            n = sum(t.count for t in items)
            t0 = min((t.first for t in items if t.first), default=None)
            t1 = max((t.last for t in items if t.last), default=None)
            lines.append(
                f"{cat}: {n:,} lines ({len(items)} kinds), {ts(t0)}-{ts(t1)}; "
                f"e.g. '{clip(items[0].key, 100)}'"
            )
        sections.append("== Routine noise, counted only ==\n" + "\n".join(lines))

    problems, used_chars = [], 0
    for t in once + repeated:
        tag = f"[{','.join(t.flags)}] " if t.flags else ""
        prog = f"{t.program}: " if t.program else ""
        line = f"{t.count}x {span(t)} {tag}{prog}{clip(t.example, 160)}"
        if used_chars + len(line) > PROBLEMS_CHARS:
            problems.append(
                f"({len(once) + len(repeated) - len(problems)} more in the digest)"
            )
            break
        problems.append(line)
        used_chars += len(line) + 1
    if dropped:
        problems.append(f"({dropped} more flagged messages not listed)")
    return Digest("\n\n".join(sections), nlines, len(templates), "\n".join(problems))


async def _summarize_log(p: Path, text: str, task: str, include_digest: bool) -> str:
    digest = build_digest(text, focus=task, name=p.name)
    system = (
        "You summarise server logs for an engineer. You are given a digest of the whole log made "
        "by a program: each line is one kind of message with its exact count, first and last "
        "time, and one real example. Be factual and concise, quote exact identifiers, times and "
        "counts from the digest, and never invent details or counts that are not in it."
    )
    prompt = (
        f"Focus: {task}\n\n{digest.text}\n\n"
        "Write the summary:\n"
        "- Start with the problems: errors, failures, hardware and power events, then warnings.\n"
        "- Mention every problem seen only 1-3 times, with its time (HH:MM:SS); one-off events "
        "matter. Lines marked 'near:' were logged seconds around the event above them; use "
        "them to tell what happened (which device, whether it came back or a recheck passed).\n"
        "- For repeating messages give the exact count and how often they recur, and group "
        "related messages (the same share, disk or device) into one point.\n"
        "- Cover routine activity and noise in at most two short lines.\n"
        "- Use the counts exactly as given (COUNTx), per message; do not merge counts of "
        "different messages into a range."
    )
    async with httpx.AsyncClient() as client:
        summary = await _chat(client, system, prompt, max_tokens=1500)
    out = summary
    if digest.problems:
        # The model can drop a time or blur a count; this list comes straight from the log.
        out += (
            "\n\nFlagged messages, exact (COUNTx FIRST-LAST [flags] program: example):\n"
            + digest.problems
        )
    out += (
        f"\n\n({p}: {digest.lines:,} log lines reduced to a {len(digest.text) / 1000:.1f} KB "
        f"digest of {digest.templates:,} distinct messages, model {CHAT_MODEL}"
        + ("" if include_digest else "; include_digest=True shows the digest")
        + ")"
    )
    if include_digest:
        out += f"\n\n{digest.text}"
    return out


async def _summarize_text(p: Path, text: str, task: str) -> str:
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
async def local_summarize(
    path: str, focus: str = "", mode: str = "auto", include_digest: bool = False
) -> str:
    """
    Summarise a large local text file (logs, syslog, diagnostics, CSV, docs) with the local model.

    Use this instead of reading a big file into context, or for files that should not leave the
    LAN. `focus` says what to look for (e.g. "errors and warnings with timestamps", "anything
    about disk sdq"). The local model is small: treat the result as a lead and verify specific
    facts in the file.

    `mode`: "auto" (default) detects timestamped logs such as syslog; "log" forces log mode;
    "text" forces the chunked text mode. In log mode a program first reduces the log to a digest
    (each kind of message once, with exact counts, first/last times and an example; errors,
    warnings, hardware events and rare lines always kept; routine noise such as sshd sessions,
    cron, disk spin-downs and docker veth changes only counted unless the focus mentions them),
    and the model summarises that digest, so counts are exact and one-off events survive.
    `include_digest=True` appends the digest itself (a few KB) for checking the summary. Other
    files are summarised in pieces, then combined.
    """
    if mode not in {"auto", "log", "text"}:
        raise ValueError(f'mode must be "auto", "log" or "text", not {mode!r}')
    p, text = _read_text(path)
    task = focus.strip() or "the main points, errors, warnings and anything unusual"
    if mode == "log" or (mode == "auto" and looks_like_log(text)):
        return await _summarize_log(p, text, task, include_digest)
    return await _summarize_text(p, text, task)


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
