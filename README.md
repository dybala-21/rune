<p align="center">
  <h1 align="center">ᚱ RUNE-BOT</h1>
  <p align="center"><strong>A local-first AI agent that learns from experience.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#what-makes-rune-different">What's Different</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a>
</p>

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white" />
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

```
❯ Fix the authentication bug in api/auth.py

  ┃  ◇ file_read api/auth.py  ✓
  ┃  ◆ file_edit api/auth.py  ✓
  ┃  ▸ bash ruff check .  ✓

✓ done — steps 1 — tools 3
  💡 learned rule applied: verify_before_edit
```

## Quick Start

```bash
# Install
curl -LsSf https://raw.githubusercontent.com/dybala-21/rune/main/install.sh | sh

# Set any LLM provider key
rune env set OPENAI_API_KEY sk-...

# Run
rune
```

Works with OpenAI, Anthropic, Gemini, Grok, Mistral, DeepSeek, Cohere, Azure, Ollama, and [100+ providers](https://docs.litellm.ai/docs/providers) via LiteLLM.

```bash
rune --message "explain the auth flow in this repo"
rune --model claude-sonnet-4-6 --provider anthropic
rune web                                    # web UI
rune self update                            # update to latest
```

## What Makes RUNE Different

**It learns from experience.** RUNE records what worked and what didn't. Successful patterns become golden experiences (✅), failures become warnings (⚠️). Next session, these are injected into context — the agent avoids past mistakes and repeats what worked. Over time, repeated failures generate prevention rules automatically.

```
## Past Experience (auto-learned)
- ✅ Fixed lint with ruff check (utility: +1)
- ⚠️ web_fetch on namu.wiki → 403 (utility: -1)

## Learned Rules
- verify_before_edit: re-read file before editing to avoid stale content
```

**It proves its work.** An Evidence Gate checks that the agent actually read the files, wrote the changes, and ran the tests. A Quality Gate catches hollow answers and error masking. If evidence is missing, the task continues — not "done."

**It asks before acting.** Every file write, every shell command goes through Guardian — pattern-based risk analysis with workspace sandboxing. You approve or deny.

**Your memory is a file you can edit.** Open `~/.rune/memory/learned.md` in any editor. Delete a line to make it forget. Everything is plain markdown.

```
~/.rune/memory/
├── MEMORY.md          # your knowledge — edit freely
├── learned.md         # auto-extracted facts + learned rules
├── daily/
│   └── 2026-03-22.md  # what happened today
└── user-profile.md    # preferences
```

## Features

### Tools

| | |
|---|---|
| **Files** | read, write, edit, delete, list, search |
| **Execution** | bash (Guardian-validated), service management |
| **Browser** | Playwright — navigate, observe, click, extract, screenshot |
| **Web** | search, fetch |
| **Code** | project map, find definitions, find references, impact analysis |
| **Memory** | search, save, tune |
| **MCP** | stdio, SSE, HTTP transports — web UI for server management |

### Multi-Channel

Same agent, same memory:

| Channel | Setup |
|---------|-------|
| **Terminal (TUI)** | `rune` |
| **Web UI** | `rune web` |
| **Telegram** | `rune env set RUNE_TELEGRAM_TOKEN <token>` |
| **Discord** | `rune env set RUNE_DISCORD_TOKEN <token>` |
| **Slack** | `rune env set RUNE_SLACK_BOT_TOKEN <token>` |

### Self-Improving

| What | How |
|------|-----|
| Episode memory | Every task result is scored (+1 golden / -1 warning) and recalled for similar future tasks |
| Behavior prediction | N-gram tool sequence prediction — suggests likely next actions |
| Rule learning | Repeated failures auto-generate prevention rules via LLM |
| Proactive suggestions | Watches workflow patterns, suggests actions, learns from dismissals |

Use `/learned` in the TUI to see learning status.

## Architecture

```
                        ┌─────────────────────┐
                        │   LLM Providers     │
                        │  OpenAI · Anthropic │
                        │  Gemini · Ollama    │
                        │  + 100 more         │
                        └─────────┬───────────┘
                                  │ LiteLLM
╔═════════════════════════════════╪════════════════════════════════╗
║  ┌──────────────────────────────┴─────────────────────────────┐  ║
║  │ INTERFACE                                                  │  ║
║  │  CLI · TUI · Web · Telegram · Discord · Slack              │  ║
║  └────────────────────────────┬───────────────────────────────┘  ║
║                               ▼                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ AGENT CORE                                                 │  ║
║  │  Agent Loop ─── Tools ─── Skills ─── MCP ─── Delegation   │  ║
║  │       │                                                    │  ║
║  │  Guardian ──── Evidence Gate ──── Quality Gate             │  ║
║  └────────────────────────┬───────────────────────────────────┘  ║
║                           ▼                                      ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ MEMORY & LEARNING                                          │  ║
║  │  Episodes (utility scoring)  ·  Rule Learner               │  ║
║  │  Behavior Predictor          ·  Proactive Engine            │  ║
║  │  FAISS vectors + markdown    ·  Code Graph (tree-sitter)    │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

## CLI

```bash
rune                              # interactive TUI
rune --message "..."              # single prompt
rune --model <model>              # specify model
rune web                          # web UI + MCP management

rune memory show                  # view memory
rune memory search <query>        # search
rune memory edit                  # open in $EDITOR
rune memory stats                 # usage stats

rune env set KEY value            # store API keys
rune self update                  # update from GitHub
rune self status                  # version info
```

## Development

```bash
git clone https://github.com/dybala-21/rune.git && cd rune
uv sync --extra dev
uv run rune                       # run from source
uv run pytest                     # tests
uv run ruff check .               # lint
```

## License

MIT — See [LICENSE](LICENSE).
