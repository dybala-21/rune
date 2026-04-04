<p align="center">
  <h1 align="center">ᚱ RUNE-BOT</h1>
  <p align="center"><strong>A local-first AI agent that learns from experience.</strong></p>
  <p align="center">Every task makes it smarter. Your data stays on your machine.</p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a>
</p>

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white" />
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

```
─── rune ──────────────────────────────────────────
  Terminal Agent · claude-sonnet · 318 episodes learned

❯ Fix the authentication bug in api/auth.py

  ┃  ◇ file_read api/auth.py  ✓
  ┃  ◆ file_edit api/auth.py  ✓
  ┃  ▸ bash ruff check .  ✓

✓ done — steps 1 — tools 3 — tokens 12k
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

Works with **OpenAI, Anthropic, Gemini, Grok, Mistral, DeepSeek, Cohere, Azure, Ollama**, and [130+ providers](https://docs.litellm.ai/docs/providers) via LiteLLM. Switch models with one config change:

```bash
rune --model claude-sonnet-4-6 --provider anthropic
rune --model gpt-4o --provider openai
rune --model gemini-2.5-flash --provider vertex_ai
```

```bash
rune                                    # interactive TUI
rune --message "explain the auth flow"  # one-shot
rune web                                # web UI
rune voice                              # voice mode (STT/TTS)
```

## How It Works

### It remembers what worked

RUNE records every task as an episode scored +1 (success) or -1 (failure). Next session, similar tasks pull from past experience. Repeated failures auto-generate prevention rules.

```
Past Experience (auto-injected into context)
  ✅ Fixed lint with ruff check (utility: +1)
  ⚠️ web_fetch on namu.wiki → 403 (utility: -1)

Learned Rules
  verify_before_edit: re-read file before editing to avoid stale content
```

### It earns your trust

Approve the same action multiple times and RUNE promotes it to auto-execute. Revert once and it demotes back. High-risk commands (sudo, rm -rf) stay manual no matter what.

### It proves its work

An Evidence Gate checks the agent actually read files, wrote changes, and ran tests. A Quality Gate catches hollow answers. If evidence is missing, the task keeps going.

### It asks before acting

Every file write, every shell command goes through Guardian — 80+ risk patterns with workspace sandboxing.

### Your memory is a file

```
~/.rune/memory/
├── MEMORY.md          # your knowledge — edit freely
├── learned.md         # auto-extracted facts + rules
├── daily/
│   └── 2026-03-22.md  # what happened today
└── user-profile.md    # preferences
```

Open in any editor. Delete a line to make it forget.

## Features

### Tools

| | |
|---|---|
| **Files** | read, write, edit, delete, list, search |
| **Execution** | bash (Guardian-validated), service management |
| **Browser** | Playwright headless — navigate, observe, click, extract, screenshot |
| **Web** | search, fetch |
| **Code** | project map, definitions, references, impact analysis (tree-sitter) |
| **Memory** | multi-source search (facts + episodes + vectors), save |
| **Voice** | STT/TTS with multi-provider auto-detection |
| **MCP** | stdio, SSE, HTTP transports — web UI for server management |

### Multi-Agent

Complex goals are decomposed into subtasks with dependency tracking:

```
╭──────┬───────────────────────────────────┬────────────────╮
│  ✓   │ Scan for security vulnerabilities │     researcher │
│  ✓   │ Fix XSS in login.py               │       executor │
│  ✓   │ Fix SQLi in query.py              │       executor │
│  ✓   │ Write security report             │       executor │
╰──────┴───────────────────────────────────┴────────────────╯
  ✓ 4/4 · 12.3s
```

- 4 roles: Researcher, Planner, Executor, Communicator — each with scoped tool access
- Independent subtasks run in parallel; dependent ones wait for upstream results
- Read-only tools run concurrently (up to 5), write tools stay serial
- Research findings can spawn follow-up tasks at runtime (dynamic DAG expansion)

### Browser Extension

RUNE can control your real Chrome browser via the **RUNE Browser Bridge** extension. This is separate from Playwright headless — it lets RUNE interact with your actual browser session (logged-in sites, cookies, etc).

```bash
# 1. Extract extension (auto-runs during install)
rune browser setup

# 2. Load in Chrome
#    Open chrome://extensions → Enable Developer mode → Load unpacked
#    Select: ~/.rune/extension/rune-browser-bridge/

# 3. Done — RUNE auto-connects when it needs the browser
```

The extension auto-discovers RUNE's relay server on `localhost:19222-19231`. No manual connection needed — when RUNE requests a browser action, the extension connects automatically.

To check connection status: `rune browser status`

### Multi-Channel

Same agent, same memory, anywhere:

| Channel | Setup |
|---------|-------|
| **Terminal (TUI)** | `rune` |
| **Web UI** | `rune web` |
| **Telegram** | `rune env set RUNE_TELEGRAM_TOKEN <token>` |
| **Discord** | `rune env set RUNE_DISCORD_TOKEN <token>` |
| **Slack** | `rune env set RUNE_SLACK_BOT_TOKEN <token>` |

### Self-Improving

| | |
|---|---|
| **Episode memory** | Every task scored +1/-1, recalled for similar future tasks |
| **Autonomy promotion** | Repeatedly approved actions auto-execute; reverts demote back |
| **Behavior prediction** | N-gram tool sequence prediction |
| **Time-slot patterns** | Learns your activity by time of day for proactive suggestions |
| **Rule learning** | Repeated failures generate prevention rules via LLM |
| **Proactive engine** | Watches patterns, suggests actions, learns from dismissals |

`/learned` in the TUI shows everything RUNE has learned.

## Architecture

```
                        ┌─────────────────────┐
                        │   LLM Providers     │
                        │  OpenAI · Anthropic │
                        │  Gemini · Ollama    │
                        │  130+ via LiteLLM   │
                        └─────────┬───────────┘
                                  │
╔═════════════════════════════════╪════════════════════════════════╗
║  ┌──────────────────────────────┴─────────────────────────────┐  ║
║  │ INTERFACE                                                  │  ║
║  │  TUI · Web · Voice · Telegram · Discord · Slack            │  ║
║  └────────────────────────────┬───────────────────────────────┘  ║
║                               ▼                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ AGENT CORE                                                 │  ║
║  │  Agent Loop ── Tools ── Skills ── MCP ── Multi-Agent       │  ║
║  │       │                                                    │  ║
║  │  Guardian ── Evidence Gate ── Quality Gate ── Autonomy     │  ║
║  └────────────────────────┬───────────────────────────────────┘  ║
║                           ▼                                      ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ MEMORY & LEARNING                                          │  ║
║  │  Episodes (utility scoring)  ·  Rule Learner               │  ║
║  │  Behavior Predictor          ·  Proactive Engine           │  ║
║  │  FAISS vectors + markdown    ·  Code Graph (tree-sitter)   │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

## LLM Configuration

```yaml
# ~/.rune/config.yaml — any one key is enough

openai_api_key: "sk-..."
anthropic_api_key: "sk-ant-..."
gemini_api_key: "AIza..."                          # Google AI Studio

# Vertex AI (service account):
google_credentials_file: "~/.rune/google-creds.json"
# project_id auto-detected from credentials file
```

## CLI

```bash
rune                              # interactive TUI
rune --message "..."              # single prompt
rune --model <model>              # specify model
rune web                          # web UI + MCP management
rune voice                        # voice mode

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
