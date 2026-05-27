# Harbor Adapter

This directory contains the RUNE adapter for Harbor/Terminal-Bench smoke runs.

Development smoke example:

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
export RUNE_HARBOR_INSTALL_MODE="auto"    # auto, wheelhouse, source, or pip
export RUNE_HARBOR_WHEELHOUSE="/rune-wheelhouse"
export RUNE_HARBOR_MODEL="gpt-5.4"
export RUNE_HARBOR_PROVIDER="openai"
export RUNE_HARBOR_MEMORY_MODE="default"  # or off
export RUNE_HARBOR_ATTEMPT_INDEX="1"
export RUNE_HARBOR_TASK_ID="adaptive-rejection-sampler"
export RUNE_HARBOR_AGENT_VARIANT_ID="rune-aa-terminal-v1"
export RUNE_HARBOR_MAX_STEPS="80"         # optional benchmark safety cap
export RUNE_HARBOR_TIMEOUT_SECONDS="900"  # optional per-attempt wall-time cap
export RUNE_HARBOR_SKIP_INSTALL="0"       # set to 1 only if the task image already has rune
export RUNE_HARBOR_PASS_ENV=""            # comma/space-separated extra env keys, if needed
```

Artifacts are written under `/logs/agent/rune` and RUNE state is isolated under
`/logs/agent/rune_home`. The installed RUNE runtime lives in
`/logs/agent/rune_venv`, and install provenance is written to
`/logs/agent/rune_install_fingerprint.json`.

The adapter only forwards credential variables for the selected provider
(`RUNE_HARBOR_PROVIDER` or `RUNE_PROVIDER`). If no provider is set, it forwards
a credential only when exactly one known provider credential is present in the
host environment. Use `RUNE_HARBOR_PASS_ENV` for any additional variables that
must be exposed to the task container.

## Official Wheelhouse Mode

For benchmark runs used in score calculations, build a read-only wheelhouse and
force the agent to install from it:

```bash
benchmarks/harbor/build_wheelhouse.sh /tmp/rune-wheelhouse
```

Build this wheelhouse on the same target platform as the benchmark task
containers, normally Linux/amd64 for Terminal-Bench. Do not use a macOS-built
wheelhouse for official Harbor runs.

Mount the wheelhouse read-only and require fingerprint fields:

```bash
mkdir -p /tmp/rune-harbor-uv-cache

harbor run \
  -d terminal-bench@2.0 \
  --include-task-name adaptive-rejection-sampler \
  --agent-env RUNE_HARBOR_TASK_ID=adaptive-rejection-sampler \
  --agent-env RUNE_HARBOR_INSTALL_MODE=wheelhouse \
  --agent-env RUNE_HARBOR_PROVIDER=openai \
  --agent-env RUNE_HARBOR_MODEL=gpt-5.4 \
  --agent-env RUNE_HARBOR_AGENT_VARIANT_ID=rune-aa-terminal-v1 \
  --agent-env RUNE_HARBOR_MAX_STEPS=80 \
  --agent-env RUNE_HARBOR_TIMEOUT_SECONDS=900 \
  --agent-env RUNE_BENCH_REQUIRE_FINGERPRINT=1 \
  --agent-env RUNE_BENCH_EXPECT_INSTALL_MODE=wheelhouse \
  --agent-env RUNE_BENCH_EXPECT_WHEELHOUSE_SHA256=<wheelhouse-manifest-sha256> \
  --mounts '[{"type":"bind","source":"/tmp/rune-wheelhouse","target":"/rune-wheelhouse","read_only":true},{"type":"bind","source":"/tmp/rune-harbor-uv-cache","target":"/uv-cache"}]' \
  --agent-import-path benchmarks.harbor.rune_agent:RuneInstalledAgent
```

The RUNE attempt artifacts include `fingerprint.json` and
`fingerprint_gate.json`. Treat any run with `fingerprint_gate.valid=false` as an
invalid benchmark attempt and rerun it after fixing provenance.

## Source Fallback Mode

For local development, mount `/rune-src`; `auto` mode installs from source when
no wheelhouse is mounted. Use this for smoke tests only; do not mix source-mode
runs into official score calculations:

```bash
mkdir -p /tmp/rune-harbor-uv-cache

harbor run \
  -d terminal-bench@2.0 \
  --include-task-name adaptive-rejection-sampler \
  --agent-env RUNE_HARBOR_TASK_ID=adaptive-rejection-sampler \
  --agent-env RUNE_HARBOR_INSTALL_MODE=source \
  --mounts '[{"type":"bind","source":"/path/to/rune","target":"/rune-src","read_only":true},{"type":"bind","source":"/tmp/rune-harbor-uv-cache","target":"/uv-cache"}]' \
  --agent-import-path benchmarks.harbor.rune_agent:RuneInstalledAgent
```
