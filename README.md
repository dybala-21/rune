<p align="center">
  <h1 align="center">ᚱ RUNE-BOT</h1>
  <p align="center"><strong>An AI agent with evidence-gated execution, editable markdown memory, and multi-provider freedom.</strong></p>
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
  <img alt="Tests" src="https://img.shields.io/badge/tests-passing-brightgreen" />
</p>

---

```
❯ Fix the authentication bug in api/auth.py

  ┃  ◇ file_read api/auth.py  ✓
  ┃  ◆ file_edit api/auth.py  ✓

╭─ rune ─────────────────────────────────────────────────────╮
│ Found the issue. The JWT expiry check uses UTC but         │
│ compares against local time.                               │
│                                                            │
│  api/auth.py                                               │
│  - expires_at = datetime.now()                             │
│  + expires_at = datetime.now(timezone.utc)                 │
│                                                            │
│  ⚠️ This modifies authentication logic. Approve? [y/n]     │
╰────────────────────────────────────────────────────────────╯
✓ Evidence Gate: read=1, write=1 — done
steps 1 — tools 2 — tokens 8.4k — cost ~$0.04 — time 3s
```

## Quick Start

```bash
# Install (no Python required — installs uv + Python automatically)
curl -LsSf https://raw.githubusercontent.com/dybala-21/rune/main/install.sh | sh

# Set any LLM provider key
rune env set OPENAI_API_KEY sk-...

# Run
rune
```

Works with OpenAI, Anthropic, Gemini, Grok, Mistral, DeepSeek, Cohere, Azure, and [100+ providers](https://docs.litellm.ai/docs/providers) via LiteLLM.

```bash
rune --message "explain the auth flow in this repo"
rune --model claude-sonnet-4-6 --provider anthropic
rune web                                    # web UI with MCP management
rune self update                            # update to latest
```

## What Makes RUNE Different

**It proves its work.** Most agents say "done" and you hope for the best. RUNE runs an Evidence Gate — 18 requirement checks that verify the agent actually read the files, wrote the changes, and ran the tests. A separate Quality Gate catches hollow answers, error masking, and unfinished drafts. If evidence is missing, the task continues.

**It asks before acting.** Every file write, every shell command, every risky operation goes through Guardian — 43 dangerous patterns, 5 risk tiers, workspace path sandboxing. You approve or deny. Over time, RUNE earns autonomy for operations you consistently approve.

**It suggests before you ask.** The Proactive Engine watches your workflow and suggests actions at the right time — but learns from rejections. After 5 "no"s on a suggestion type, it raises the threshold. Quiet hours are respected.

**It budgets every token.** 4-phase execution (thinking → acting → reflecting → wind-down) with intent-based budgets — 50K for chat, up to 1M for complex tasks. Stall detection catches loops before they burn tokens. When stuck, it admits it.

**Your memory is a file you can edit.** Open `~/.rune/memory/MEMORY.md` in any editor. Delete a line to make it forget. Commit it to Git. The vector index is a cache — rebuild from markdown anytime.

```
~/.rune/memory/
├── MEMORY.md          # your knowledge — edit freely
├── learned.md         # auto-extracted facts
├── daily/
│   └── 2026-03-20.md  # what happened today
└── user-profile.md    # preferences and stats
```

## Features

### 49 Built-in Tools

| | |
|---|---|
| **Files** | read, write, edit, delete, list, search |
| **Execution** | bash (Guardian-validated), service management |
| **Browser** | 10 Playwright tools — navigate, observe, click, extract, screenshot |
| **Web** | search, fetch with adaptive domain failure learning |
| **Code** | project map, find definitions, find references, impact analysis |
| **Memory** | search, save, tune — 13 CLI commands for inspection and maintenance |
| **Scheduling** | cron create, list, update, delete |
| **Delegation** | sub-agent tasks, multi-task orchestration |
| **MCP** | 3 transports (stdio, SSE, HTTP), web UI for server management |

---

### Code Intelligence

Tree-sitter parsing across 173 languages. Repo map auto-selects the most-referenced symbols and injects them into agent context within a token budget. Incremental — only changed files are re-parsed.

---

### Multi-Channel

Same agent, same memory, same approval flow:

| Channel | Setup |
|---------|-------|
| **Terminal (TUI)** | included |
| **Web UI** | `rune web` |
| **Telegram** | `rune env set RUNE_TELEGRAM_TOKEN <token>` |
| **Discord** | `rune env set RUNE_DISCORD_TOKEN <token>` |
| **Slack** | `rune env set RUNE_SLACK_BOT_TOKEN <token>` |
| **LINE, WhatsApp, Google Chat, Mattermost** | from source |

Set a token and channels auto-start with the TUI.

---

### Skills & Scheduling

Reusable workflows as `SKILL.md` files. Personal (`~/.rune/skills/`) or project-scoped (`.rune/skills/`). Cron-based scheduling for recurring agent tasks.

## Architecture

```
                        ┌─────────────────────┐
                        │   LLM Providers     │
                        │  OpenAI · Anthropic │
                        │  Gemini · Grok      │
                        │  Mistral · DeepSeek │
                        │  Cohere · Azure     │
                        │  Ollama · 100+ more │
                        └─────────┬───────────┘
                                  │ LiteLLM
╔═════════════════════════════════╪════════════════════════════════╗
║  ┌──────────────────────────────┴─────────────────────────────┐  ║
║  │ INTERFACE                                                  │  ║
║  │  CLI · TUI · Web · Telegram · Discord · Slack · more       │  ║
║  └────────────────────────────┬───────────────────────────────┘  ║
║                               ▼                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ AGENT CORE                                                 │  ║
║  │                                                            │  ║
║  │  Agent Loop ─── 49 Tools ─── Skills ─── MCP ─── Delegation │  ║
║  │       │                                                    │  ║
║  │  Guardian ──── Evidence Gate ──── Quality Gate             │  ║
║  └────────────────────────┬───────────────────────────────────┘  ║
║                           ▼                                      ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ INTELLIGENCE                                               │  ║
║  │                                                            │  ║
║  │  Memory (Markdown + FAISS)  ·  Proactive Engine            │  ║
║  │  Code Graph (tree-sitter)   ·  Repo Map (ranked context)   │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

<details>
<summary><strong>Module overview</strong></summary>

| Module | What it does |
|--------|-------------|
| `agent/` | Core loop, token budgeting, evidence gate, quality gate, stall detection, failover, cognitive cache |
| `safety/` | Guardian (43 patterns), dual-pass analysis, path sandboxing, adaptive autonomy |
| `memory/` | Markdown store, RRF search, FAISS vector cache, 13 CLI commands |
| `intelligence/` | Tree-sitter code graph (173 languages), repo map |
| `proactive/` | 8-step suggestion pipeline, reflexion learner |
| `capabilities/` | 49 tools with adaptive fetch failure learning |
| `channels/` | TUI, Web, Telegram, Discord, Slack, LINE, WhatsApp, Google Chat, Mattermost |
| `mcp/` | Multi-server MCP client, web-based server management |
| `skills/` | SKILL.md executor |
| `api/` | FastAPI server, SSE/WebSocket/NDJSON streaming |
| `llm/` | LiteLLM adapter, 8 providers, 71 models |

</details>

## Configuration

```yaml
# ~/.rune/config.yaml
llm:
  defaultProvider: openai
  activeProvider: anthropic       # set by /model, persisted
  activeModel: claude-sonnet-4-6

guardian:
  autonomy_level: supervised      # supervised | semi-autonomous | autonomous

proactive:
  enabled: true
  quiet_hours: [22, 8]
```

## CLI

```bash
# Agent
rune                              # interactive TUI (channels auto-start)
rune --message "..."              # single prompt, non-interactive
rune --model <model>              # specify model
rune web                          # web UI + MCP server management

# Memory
rune memory show                  # view current memory
rune memory search <query>        # semantic + keyword search
rune memory edit                  # open in $EDITOR
rune memory forget <key>          # delete and suppress re-extraction
rune memory stats                 # usage, hit counts, gc candidates

# Configuration
rune env set KEY value            # store in ~/.rune/.env
rune env list                     # view all configured keys

# Management
rune self update                  # reinstall from GitHub
rune self uninstall               # remove RUNE
rune self status                  # version, installer, daemon info
```

Inside the TUI, type `/` for slash commands: `/model`, `/memory`, `/help`, `/undo`, `/diff`, `/export`, and more.

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
