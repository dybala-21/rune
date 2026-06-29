# Weak-model deliverability benchmark

**Question:** on a weak *local* model, does RUNE's harness make the model actually
**produce a verifiable deliverable** for a multi-step coding task — versus emitting
a plan and no files? **Answer (measured): only with guided decoding on — then 100%,
all correct; without it, 0%.**

Each task asks an agent to write a small module + tests and run them. Scoring is
done by **hidden ground-truth verifiers** (`tasks.py`) that check correctness
traps directly (population vs sample stdev, even-length median, tie-breaks, edge
cases). The agent's own tests are ignored, so it can't pass with lenient tests.

```
python benchmarks/weak_model_deliverability/run.py --model qwen2.5-coder:7b \
    --rune-bin .venv/bin/rune
```

The same runner + verifier scores **any** agent, so head-to-head is reproducible:

```
# Hermes Agent (or any OpenAI-compatible agent) via a command template:
python benchmarks/weak_model_deliverability/run.py \
  --agent-cmd 'python /path/to/hermes/run_agent.py --query {task} \
     --model qwen2.5-coder:7b --base_url http://localhost:11434/v1 \
     --api_key ollama --max_turns 25'
```

## What we measured (2026-06-29, `qwen2.5-coder:7b`)

**The decisive variable is guided decoding** (`RUNE_GUIDED_TOOLS`, default OFF):
RUNE grammar-constrains each turn to a valid tool-call schema for local models, so
a weak model can't just emit a prose plan. Same task, same model, fixed runner.

**Two tasks, and they disagree — which is the honest finding:**

| Task | guided OFF (tool / artifact) | guided ON (tool / artifact / correct) |
|---|---|---|
| `stats`    | 0/6 / **0/6** | 10/10 / **10/10** / **10/10 (6/6)** |
| `wordfreq` | 0/5 / **0/5** | **5/5** / **0/5** / 0/5 |

Head-to-head, same fixed runner & model: **Hermes** (no guided decoding) on `stats`
= **0/5 artifact** — so the deliverability is RUNE's harness, not the model. Strong
model `qwen3-coder:480b-cloud`: RUNE and Hermes both 6/6 — a tie; a capable model
doesn't need the constraint.

### Conclusion — the edge is real but BOUNDED by the model's capability
- **`stats`: guided decoding turns 0 → 100%, all correct.** The 7B writes valid,
  correct code; OFF it emits a markdown plan and never calls a tool. The artifact
  was inspected by hand and scores 6/6 on the hidden traps (population stdev, mode
  tie→smallest, empty→ValueError) — it is not gaming the verifier. This is the real
  harness effect, and Hermes (measured) gets the 0 column.
- **`wordfreq`: guided decoding fixes tool-*calling* (0/5 → 5/5) but NOT delivery
  (still 0/5).** The 7B is forced to act, calls `file_write`, but the code it
  generates is *syntactically invalid* (`ast.parse` fails — an unbalanced paren in
  the punctuation-strip expression). RUNE's `syntax_guard` correctly rejects the
  write, the model regenerates broken code again, and it never lands a valid file.
  This is a **capability ceiling of the 7B**, not a harness control-flow problem —
  guided decoding cannot make a model emit code it cannot write.
- **So "0 → 100% is the whole harness effect" was a `stats`-only overgeneralization.**
  The honest claim: *guided decoding makes a weak local model reliably DELIVER for
  tasks within its own code-generation capability; it cannot exceed that ceiling.*
  Harness fixes PROCESS failures, not CAPABILITY failures.
- **Defect found & fixed along the way (litellm_adapter.py):** before, after a
  syntax-rejected write the guided schema still offered a `{final}` branch, so the
  weak model bailed out with a *prose* "please fix it" answer and the loop ended
  with 0 files (a quiet near-false-completion). Fix: while no real artifact has
  succeeded, the guided schema drops the `{final}` branch entirely (capped at
  `_GUIDED_FORCE_ACTION_CAP`), forcing a tool retry instead of a prose escape.
  Measured effect: behavior changed from an instant ~38 s bail to sustained retries
  that fail *honestly* ("syntax error, manual fix needed") — it does **not** rescue
  `wordfreq` (capability-floored), and `stats` stays 3/3 (no regression).
- **Caveat: guided decoding is opt-in (`RUNE_GUIDED_TOOLS`, default OFF)** — but is
  now defaulted ON for local ollama models (gating excludes cloud/strong), so the
  out-of-box weak-model experience gets the working column for in-capability tasks.

### How this number was hard-won (and why it's trustworthy)
The first runs of this benchmark reported **0/20 even with the lever effectively
off**, and an early "fix" still showed 0 — because the *runner itself* had a bug:
it ran each task in a macOS `tempfile` dir under `/var/folders`, where the agent's
`file_write` (which `.resolve()`s paths) wrote real files that the runner then
failed to read back, scoring a correct artifact as 0. Switching workspaces to
`/tmp` fixed it. The lesson cuts both ways: a benchmark can produce false negatives
about the very thing it measures — only the adversarial check (a manual run made
files; the automated run didn't) surfaced it. And a *second* task (`wordfreq`)
caught a *second* overclaim — that the `stats` 0→100% was the general weak-model
effect — by failing where `stats` passed (capability ceiling). Raw result JSON is
checked in: `results_guided_{off_n6,on_n10}.json` (stats),
`results_hermes_stats_n5.json` (head-to-head),
`results_wordfreq_guided_{off,on}_n5.json` (the capability-ceiling counter-example).

## Reproduce
```
# the A/B above — guided decoding OFF vs ON
RUNE_GUIDED_TOOLS=0 python benchmarks/weak_model_deliverability/run.py --trials 6
RUNE_GUIDED_TOOLS=1 python benchmarks/weak_model_deliverability/run.py --trials 10
```
Each run scores the produced files with the hidden verifiers, in a fresh isolated
(cold) `RUNE_HOME` per trial. Weak-model runs are noisy; use several `--trials`.
