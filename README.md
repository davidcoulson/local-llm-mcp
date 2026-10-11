# local-llm MCP server

Lets Claude Code hand bulk or private text work to local models in Ollama on the Unraid server's
RTX 3070 (`http://10.2.6.50:11434`): `local_summarize` (big logs/diagnostics), `local_ask`, and
`semantic_search` (meaning-based search over a folder, cached embeddings in `~/.cache/local-llm-mcp`).

Registered for Claude Code (user scope) on the Mac and in the `claude-code` container:

    claude mcp add --scope user local-llm -- uv run --script /path/to/server.py

The models are small (qwen3:8b): treat results as leads and verify in the files.

## Log mode

`local_summarize` detects timestamped logs (syslog, ISO-timestamped app logs; or force it with
`mode="log"`, and `mode="text"` for the old chunked path). A deterministic pre-pass reduces the
log to a digest of a few KB: each kind of message once (numbers, IPs, hex, IDs masked) with its
exact count, first/last time, period if it recurs steadily, and one real example. Errors,
warnings, failures and hardware/power events are always listed, one-off problems with the rare
lines logged seconds around them; routine noise (sshd sessions, cron, disk spin-downs, docker
veth churn) is only counted unless `focus` mentions it. The model summarises the digest in one
call, and the tool appends the exact flagged-message list; `include_digest=True` returns the
whole digest too.

Tests: `uv run --with pytest --with "mcp>=1.20,<2" --with httpx --with numpy pytest`
