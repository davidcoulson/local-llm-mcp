# local-llm MCP server

An MCP server that lets Claude Code hand bulk or private text work to small local models in
[Ollama](https://ollama.com), so big logs and private files stay on your network and out of
Claude's context:

- `local_summarize`: summaries of big logs, diagnostics dumps and long tool output (with a log mode,
  below).
- `local_ask`: a quick question about a file or a piece of text.
- `semantic_search`: meaning-based search over a folder, with embeddings cached in
  `~/.cache/local-llm-mcp`.

The models are small (qwen3:8b by default): treat results as leads and verify them in the files.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and an Ollama server with the two models pulled:

    ollama pull qwen3:8b
    ollama pull qwen3-embedding:0.6b

Register it with Claude Code (user scope), pointing `OLLAMA_URL` at your Ollama server:

    claude mcp add --scope user local-llm -e OLLAMA_URL=http://192.168.1.50:11434 -- uv run --script /path/to/server.py

Configuration (environment variables):

| Variable | Default |
| --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` |
| `LOCAL_CHAT_MODEL` | `qwen3:8b` |
| `LOCAL_EMBED_MODEL` | `qwen3-embedding:0.6b` |
| `LOCAL_LLM_CACHE` | `~/.cache/local-llm-mcp` |

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

## License

MIT, see [LICENSE](LICENSE).
