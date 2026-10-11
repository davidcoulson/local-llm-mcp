# local-llm MCP server

Lets Claude Code hand bulk or private text work to local models in Ollama on the Unraid server's
RTX 3070 (`http://10.2.6.50:11434`): `local_summarize` (big logs/diagnostics), `local_ask`, and
`semantic_search` (meaning-based search over a folder, cached embeddings in `~/.cache/local-llm-mcp`).

Registered for Claude Code (user scope) on the Mac and in the `claude-code` container:

    claude mcp add --scope user local-llm -- uv run --script /path/to/server.py

The models are small (qwen3:8b): treat results as leads and verify in the files.
