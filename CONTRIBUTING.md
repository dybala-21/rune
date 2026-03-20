# Contributing to RUNE

Thanks for your interest in contributing. Here's how to get started.

## Setup

```bash
git clone https://github.com/dybala-21/rune.git
cd rune
uv sync --extra dev
```

## Development Workflow

```bash
# Run tests (must pass before submitting)
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy rune/
```

## Code Style

- **Formatter**: ruff (line-length 100)
- **Type hints**: All public functions should have type annotations
- **Imports**: Sorted by ruff isort. Heavy dependencies use lazy imports inside functions for startup performance
- **Error handling**: Use `log.warning()` or `log.debug()` instead of bare `except: pass`
- **Naming**: snake_case for functions/variables, PascalCase for classes

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure `pytest`, `ruff check`, and `mypy` all pass
4. Open a PR with a description of **what** changed and **why**

## What to Contribute

- Bug fixes with a reproducing test case
- Performance improvements with benchmarks
- New tool capabilities (add to `rune/capabilities/`)
- Channel adapters (add to `rune/channels/`)
- Documentation improvements
- Test coverage for untested modules

## What to Avoid

- Large refactors without prior discussion (open an issue first)
- Adding dependencies to core (use optional extras)
- Breaking changes to the CLI interface
- Commits with `--no-verify` or `--force`

## Project Structure

```
rune/
  agent/          Core agent loop, tool adapter, prompts
  safety/         Guardian, analyzer, execution policy
  memory/         SQLite store, FAISS vector, consolidation
  capabilities/   Tool implementations (file, bash, browser, web, ...)
  channels/       Chat adapters (Telegram, Discord, Slack, ...)
  daemon/         Background daemon, WebSocket gateway
  api/            FastAPI server, handlers
  ui/             Terminal UI (Textual + Rich)
  cli/            Typer CLI entry point
  proactive/      Suggestion engine, reflexion learner
  intelligence/   Code graph (tree-sitter)
  conversation/   Multi-turn context management
  mcp/            Model Context Protocol client
  skills/         SKILL.md executor
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
