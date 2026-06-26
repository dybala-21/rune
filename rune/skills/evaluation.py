"""Gated Skill Learning — evaluation & promotion decision (T1-1).

Given trial/success counts for a skill (with vs without injection), decide
whether the skill should be PROMOTED (measurably helps), REJECTED (regresses),
or HELD (not enough evidence yet).

The decision is Bayesian (Beta-Binomial) so it is robust to repeated peeking —
unlike a fixed p-value threshold checked every time new data arrives.

Two analysis modes:

* **unpaired** — independent treatment/control arms (online interleaving).
  ``p_with ~ Beta(1+s, 1+f)``, ``p_without ~ Beta(1+s, 1+f)``;
  ``lift = p_with - p_without`` estimated by Monte-Carlo.
* **paired** — same task run with and without the skill (offline replay).
  Only discordant pairs carry signal (McNemar). ``b`` = with-success/
  without-fail, ``c`` = with-fail/without-success. With discordant fraction
  ``f = (b+c)/n`` and ``θ ~ Beta(1+b, 1+c)``, the absolute lift is
  ``f·(2θ−1)``. Paired removes per-task difficulty variance, so it needs far
  fewer samples — preferred when a replay corpus exists.

Pure stdlib: Monte-Carlo via ``random.betavariate`` (no numpy dependency).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

# Lifecycle actions returned by the decision functions.
PROMOTE: Final = "promote"
REJECT: Final = "reject"
HOLD: Final = "hold"

_DEFAULT_MC_SAMPLES: Final = 20_000


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Promotion thresholds (mirrors SkillsConfig fields)."""

    delta_min: float = 0.05          # minimum meaningful absolute lift
    # Required posterior confidence (π). 0.95 chosen via a power/safety sweep:
    # vs 0.90 it cuts false promotion of useless skills ~4%->~1% while keeping
    # strong-effect detection high (~82% @ lift +0.30, n=48).
    prob_threshold: float = 0.95
    min_samples_paired: int = 12     # min discordant pairs for a paired call
    min_samples_online: int = 40     # min trials per arm for an unpaired call
    mc_samples: int = _DEFAULT_MC_SAMPLES


@dataclass(frozen=True, slots=True)
class EvalDecision:
    """Outcome of an evaluation."""

    action: str               # PROMOTE | REJECT | HOLD
    observed_lift: float      # point estimate of the success-rate lift
    prob_positive: float      # P(lift > δ_min)
    prob_negative: float      # P(lift < 0)
    n: int                    # effective sample size used
    reason: str


def _rng(cfg: EvalConfig, seed: int | None) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random()


def decide_unpaired(
    with_n: int, with_s: int,
    without_n: int, without_s: int,
    cfg: EvalConfig | None = None,
    *, seed: int | None = None,
) -> EvalDecision:
    """Decide from two independent arms (online interleaving)."""
    cfg = cfg or EvalConfig()
    n = min(with_n, without_n)
    if with_n == 0 or without_n == 0:
        return EvalDecision(HOLD, 0.0, 0.0, 0.0, n, "missing one arm")

    p_with = with_s / with_n
    p_without = without_s / without_n
    observed = p_with - p_without

    rng = _rng(cfg, seed)
    a_w, b_w = 1 + with_s, 1 + (with_n - with_s)
    a_o, b_o = 1 + without_s, 1 + (without_n - without_s)
    pos = neg = 0
    for _ in range(cfg.mc_samples):
        lift = rng.betavariate(a_w, b_w) - rng.betavariate(a_o, b_o)
        if lift > cfg.delta_min:
            pos += 1
        if lift < 0:
            neg += 1
    prob_pos = pos / cfg.mc_samples
    prob_neg = neg / cfg.mc_samples

    return _verdict(cfg, observed, prob_pos, prob_neg, n,
                    enough=n >= cfg.min_samples_online, mode="unpaired")


def decide_paired(
    b: int, c: int, n: int,
    cfg: EvalConfig | None = None,
    *, seed: int | None = None,
) -> EvalDecision:
    """Decide from paired replay outcomes.

    ``b`` = pairs where the skill helped (with✓, without✗);
    ``c`` = pairs where it hurt (with✗, without✓);
    ``n`` = total pairs run.
    """
    cfg = cfg or EvalConfig()
    discordant = b + c
    if n <= 0 or discordant == 0:
        return EvalDecision(HOLD, 0.0, 0.0, 0.0, discordant,
                            "no discordant pairs")

    f = discordant / n
    observed = (b - c) / n

    rng = _rng(cfg, seed)
    a, beta = 1 + b, 1 + c
    pos = neg = 0
    for _ in range(cfg.mc_samples):
        theta = rng.betavariate(a, beta)
        lift = f * (2.0 * theta - 1.0)
        if lift > cfg.delta_min:
            pos += 1
        if lift < 0:
            neg += 1
    prob_pos = pos / cfg.mc_samples
    prob_neg = neg / cfg.mc_samples

    return _verdict(cfg, observed, prob_pos, prob_neg, discordant,
                    enough=discordant >= cfg.min_samples_paired, mode="paired")


def _verdict(
    cfg: EvalConfig, observed: float, prob_pos: float, prob_neg: float,
    n: int, *, enough: bool, mode: str,
) -> EvalDecision:
    # Regression is decided even on smaller samples (fail-closed): a skill that
    # looks harmful should not linger waiting for n_min.
    if prob_neg >= cfg.prob_threshold:
        return EvalDecision(REJECT, observed, prob_pos, prob_neg, n,
                            f"{mode}: P(lift<0)={prob_neg:.2f} ≥ π")
    if enough and prob_pos >= cfg.prob_threshold:
        return EvalDecision(PROMOTE, observed, prob_pos, prob_neg, n,
                            f"{mode}: P(lift>δ)={prob_pos:.2f} ≥ π, n={n}")
    if not enough:
        return EvalDecision(HOLD, observed, prob_pos, prob_neg, n,
                            f"{mode}: need more samples (n={n})")
    return EvalDecision(HOLD, observed, prob_pos, prob_neg, n,
                        f"{mode}: inconclusive (P(lift>δ)={prob_pos:.2f})")


def eval_config_from_settings(skills_cfg: object) -> EvalConfig:
    """Build an EvalConfig from a SkillsConfig (or any object with the fields)."""
    return EvalConfig(
        delta_min=getattr(skills_cfg, "eval_delta_min", 0.05),
        prob_threshold=getattr(skills_cfg, "eval_prob_threshold", 0.95),
        min_samples_paired=getattr(skills_cfg, "eval_min_samples_paired", 12),
        min_samples_online=getattr(skills_cfg, "eval_min_samples_online", 40),
    )


# Orchestration: read counts -> decide -> transition skill state.

@dataclass(frozen=True, slots=True)
class EvalReport:
    """Result of evaluating one skill (decision + applied state change)."""

    skill_name: str
    decision: EvalDecision
    old_state: str
    new_state: str


def evaluate_skill_counts(
    *,
    paired: dict[str, int] | None,
    unpaired: dict[str, int] | None,
    cfg: EvalConfig | None = None,
    seed: int | None = None,
) -> EvalDecision:
    """Decide from whichever data exists, preferring paired (more powerful).

    ``paired``  : ``{"b", "c", "n"}`` from ``get_skill_paired_counts``.
    ``unpaired``: ``{"with_n","with_s","without_n","without_s"}`` summary.
    """
    cfg = cfg or EvalConfig()
    if paired and paired.get("n", 0) > 0:
        return decide_paired(paired["b"], paired["c"], paired["n"],
                             cfg, seed=seed)
    if unpaired:
        return decide_unpaired(
            unpaired.get("with_n", 0), unpaired.get("with_s", 0),
            unpaired.get("without_n", 0), unpaired.get("without_s", 0),
            cfg, seed=seed,
        )
    return EvalDecision(HOLD, 0.0, 0.0, 0.0, 0, "no data")


class SkillEvaluator:
    """Evaluate distilled skills and drive their lifecycle transitions.

    Pulls trial/success counts from the store, decides, and applies the
    resulting state transition in-memory. It relies on paired control data
    produced by the replay runner; with no control data a candidate simply
    HOLDs (never promoted on one-armed evidence).
    """

    def __init__(self, store: object, registry: object,
                 cfg: EvalConfig | None = None) -> None:
        self._store = store
        self._registry = registry
        self._cfg = cfg or EvalConfig()

    def evaluate(self, skill_name: str, *, seed: int | None = None) -> EvalDecision:
        paired = None
        unpaired = None
        try:
            paired = self._store.get_skill_paired_counts(skill_name)
        except Exception:
            paired = None
        try:
            unpaired = self._store.get_skill_eval_summary(skill_name)
        except Exception:
            unpaired = None
        return evaluate_skill_counts(
            paired=paired, unpaired=unpaired, cfg=self._cfg, seed=seed,
        )

    def evaluate_and_transition(
        self, skill: object, *, seed: int | None = None,
    ) -> EvalReport:
        from rune.skills.lifecycle import get_state, next_state, set_state

        decision = self.evaluate(skill.name, seed=seed)
        old = get_state(skill)
        new = next_state(old, decision.action)
        if new != old:
            set_state(skill, new)
        return EvalReport(skill.name, decision, old, new)

    def run_cycle(self, *, seed: int | None = None) -> list[EvalReport]:
        """Evaluate every candidate/shadow/active skill and apply transitions."""
        from rune.skills.lifecycle import SkillState, get_state

        evaluable = {SkillState.CANDIDATE, SkillState.SHADOW, SkillState.ACTIVE}
        reports: list[EvalReport] = []
        for skill in self._registry.list():
            if get_state(skill) in evaluable:
                reports.append(self.evaluate_and_transition(skill, seed=seed))
        return reports
