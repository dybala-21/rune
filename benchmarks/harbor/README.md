# Harbor Adapter

This directory contains the RUNE adapter for Harbor/Terminal-Bench smoke runs.

Example:

```bash
harbor run \
  -d terminal-bench@2.0 \
  --include-task-name adaptive-rejection-sampler \
  --agent-env RUNE_HARBOR_TASK_ID=adaptive-rejection-sampler \
  --agent-import-path benchmarks.harbor.rune_agent:RuneInstalledAgent
```

The adapter runs RUNE inside the task container through:

```bash
rune bench run --benchmark terminal-bench-v2 --task-id <task> --attempt-index <n> --instruction <prompt>
```

Useful environment overrides:

```bash
export RUNE_HARBOR_INSTALL_CMD="python3 -m pip install --user rune-ai"
export RUNE_HARBOR_MODEL="gpt-5.4"
export RUNE_HARBOR_PROVIDER="openai"
export RUNE_HARBOR_MEMORY_MODE="default"  # or off
export RUNE_HARBOR_ATTEMPT_INDEX="1"
export RUNE_HARBOR_TASK_ID="adaptive-rejection-sampler"
```

Artifacts are written under `/logs/agent/rune` and RUNE state is isolated under
`/logs/agent/rune_home`.
