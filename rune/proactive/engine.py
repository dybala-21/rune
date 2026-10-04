"""Generate, rank, deduplicate and persist proactive suggestions."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.memory.store import MemoryStore

log = get_logger(__name__)

_engine: ProactiveEngine | None = None

_MAX_SEEN_IDS = 10_000

# Deduplication cooldown per title key (seconds)
_DEDUP_COOLDOWN_SECS = 300  # 5 minutes

# Human-readable descriptions for inferred needs
_NEED_DESCRIPTIONS: dict[str, str] = {
    "documentation": "You've been reading extensively without writing — you may need documentation or reference material.",
    "testing": "Multiple edits without test runs detected — consider adding or running tests.",
    "refactoring": "Edits across many files detected — consider refactoring for consistency.",
}


class ProactiveEngine:
    """Manage suggestions, feedback, cooldowns and event listeners."""

    __slots__ = (
        "_config",
        "_feedback",
        "_seen_ids",
        "_suggestions",
        "_recent_suggestion_keys",
        "_dismissed_keys",
        "_evaluation_count",
        "_listeners",
        "_store",
        "_store_version",
    )

    _DISMISS_COOLDOWN_SECS = 1800  # 30 min cooldown for dismissed suggestions

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._feedback: dict[str, bool] = {}  # suggestion_id -> accepted
        self._seen_ids: set[str] = set()
        self._suggestions: dict[str, Suggestion] = {}  # id -> Suggestion
        self._recent_suggestion_keys: dict[str, datetime] = {}  # title_key -> last_added
        self._dismissed_keys: dict[str, datetime] = {}  # title_key -> dismissed_at
        self._evaluation_count: int = 0
        self._listeners: dict[str, list[Any]] = {}  # event_name -> [callbacks]
        self._store: MemoryStore | None = None
        self._store_version: tuple[int, int] | None = None

    # Event emitter

    def on(self, event: str, callback: Any) -> None:
        """Register a listener for suggestion, intervention, decision or task events."""
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Any) -> None:
        """Remove a listener for an event."""
        if event in self._listeners:
            with contextlib.suppress(ValueError):
                self._listeners[event].remove(callback)

    def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners."""
        for cb in self._listeners.get(event, []):
            try:
                cb(*args)
            except Exception as exc:
                log.warning("event_listener_error", event_name=event, error=str(exc))

    def emit_task_completed(self, goal: str, result: dict[str, Any] | None = None) -> None:
        """Emit a task_completed event (called by external systems)."""
        self._emit("task_completed", goal, result or {})

    def emit_task_failed(self, goal: str, error: str = "") -> None:
        """Emit a task_failed event (called by external systems)."""
        self._emit("task_failed", goal, error)

    def emit_context_switch(self, context: dict[str, Any]) -> None:
        """Emit a context_switch event (called by external systems)."""
        self._emit("context_switch", context)

    # Public API - Evaluation pipeline

    async def evaluate(self, context: dict[str, Any]) -> list[Suggestion]:
        """Gather, filter, rank and deduplicate suggestions within configured limits."""
        self._evaluation_count += 1

        enriched = await self._gather_context(context)

        if enriched.get("suppress", False):
            return []

        candidates = await self._generate_candidates(enriched)

        candidates = self._filter_candidates(candidates)

        candidates = self._rank_candidates(candidates)

        candidates = self._deduplicate(candidates)

        max_suggestions = self._config.get("max_suggestions", 3)
        candidates = candidates[:max_suggestions]

        for s in candidates:
            self._persist(s)
            self._seen_ids.add(s.id)
            self._suggestions[s.id] = s
        if len(self._seen_ids) > _MAX_SEEN_IDS:
            # Keep this cache bounded; durable state still prevents replay.
            to_remove = list(self._seen_ids)[: _MAX_SEEN_IDS // 2]
            for item in to_remove:
                self._seen_ids.discard(item)

        log.debug("proactive_evaluated", count=len(candidates))

        if candidates:
            self._emit("suggestion", candidates)
            # Check for intervention-level suggestions (high confidence)
            interventions = [s for s in candidates if s.confidence >= 0.8]
            if interventions:
                self._emit("intervention", interventions)
            self._emit("decision", {
                "evaluation_count": self._evaluation_count,
                "suggestions": len(candidates),
                "interventions": len(interventions) if candidates else 0,
            })

        return candidates

    # Public API - Suggestion CRUD

    def add_suggestion(self, suggestion: Suggestion) -> None:
        """Add a suggestion unless its title is still within the deduplication cooldown."""
        # Dedup cooldown check
        title_key = suggestion.title.lower().strip()
        now = datetime.now(UTC)

        # Check dismiss cooldown (30 min) first, then normal dedup (5 min)
        if title_key and title_key in self._dismissed_keys:
            dismissed_at = self._dismissed_keys[title_key]
            if (now - dismissed_at).total_seconds() < self._DISMISS_COOLDOWN_SECS:
                log.debug("suggestion_dismiss_cooldown", title=suggestion.title)
                return
            del self._dismissed_keys[title_key]  # cooldown expired

        if title_key and title_key in self._recent_suggestion_keys:
            last_added = self._recent_suggestion_keys[title_key]
            if (now - last_added).total_seconds() < _DEDUP_COOLDOWN_SECS:
                log.debug("suggestion_dedup_cooldown", title=suggestion.title)
                return

        self._recent_suggestion_keys[title_key] = now
        self._persist(suggestion)
        self._suggestions[suggestion.id] = suggestion

        # Prune stale cooldown entries (keep at most 200)
        if len(self._recent_suggestion_keys) > 200:
            cutoff = now - timedelta(seconds=_DEDUP_COOLDOWN_SECS)
            self._recent_suggestion_keys = {
                k: v
                for k, v in self._recent_suggestion_keys.items()
                if v > cutoff
            }
        # Prune stale dismiss entries
        if len(self._dismissed_keys) > 100:
            dismiss_cutoff = now - timedelta(seconds=self._DISMISS_COOLDOWN_SECS)
            self._dismissed_keys = {
                k: v
                for k, v in self._dismissed_keys.items()
                if v > dismiss_cutoff
            }

        log.debug("suggestion_added", id=suggestion.id)

    def get_suggestion(self, suggestion_id: str) -> Suggestion | None:
        """Fetch a suggestion by ID."""
        return self._suggestions.get(suggestion_id)

    def get_first_pending(self) -> Suggestion | None:
        """Return the oldest eligible unprocessed suggestion, or None."""
        min_confidence = self._config.get("min_confidence", 0.2)
        now = datetime.now(UTC)

        # Iterate in insertion order (oldest first in CPython 3.7+)
        for s in self._suggestions.values():
            if s.status != "pending" or s.execution_status is not None:
                continue
            if s.confidence < min_confidence:
                continue
            if s.expires_at and s.expires_at < now:
                continue
            return s
        return None

    def delete_suggestion(self, suggestion_id: str) -> None:
        """Remove a suggestion by ID."""
        if self._store is not None:
            from rune.proactive.state import expire
            expire(self._store, suggestion_id)
        self._suggestions.pop(suggestion_id, None)

    def handle_response(self, suggestion_id: str, accepted: bool) -> bool:
        """Record an explicit response, once, without treating it as execution."""
        suggestion = self._suggestions.get(suggestion_id)
        desired = "accepted" if accepted else "dismissed"
        if self._store is not None:
            from rune.proactive.state import respond
            suggestion = respond(self._store, suggestion_id, accepted)
        elif suggestion is not None:
            if suggestion.status not in ("pending", desired):
                raise ValueError("This suggestion already has a different response")
            if suggestion.expires_at and suggestion.expires_at <= datetime.now(UTC):
                raise ValueError("This suggestion has expired")
            suggestion.status = desired
            suggestion.response_source = "user"
        if suggestion is None:
            return False
        self._suggestions[suggestion_id] = suggestion
        self._feedback[suggestion_id] = accepted
        if not accepted:
            self._dismissed_keys[suggestion.title.lower().strip()] = datetime.now(UTC)

        log.debug(
            "proactive_feedback",
            suggestion_id=suggestion_id,
            accepted=accepted,
        )
        return True

    def _persist(self, suggestion: Suggestion) -> None:
        if self._store is not None:
            from rune.proactive.state import save
            save(self._store, suggestion)

    def record_execution(self, suggestion_id: str, status: str, result: dict | None = None) -> None:
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion is not None:
            suggestion.execution_status = status
            suggestion.execution_result = result or {}
            self._persist(suggestion)

    def list_suggestions(self) -> list[Suggestion]:
        if self._store is not None and (
            self._store.conn.pragma("data_version"), self._store.conn.total_changes()
        ) != self._store_version:
            self.load_persisted_suggestions(self._store)
        return list(self._suggestions.values())

    # Public API - Persistence

    def load_persisted_suggestions(self, store: MemoryStore) -> int:
        """Load persisted suggestions and return the count."""
        from rune.proactive.state import restore

        self._store = store
        self._store_version = (store.conn.pragma("data_version"), store.conn.total_changes())
        rows = store.get_suggestion_state()
        loaded = 0
        seen: set[str] = set()
        for row in rows:
            try:
                suggestion = restore(row)
            except (TypeError, ValueError, KeyError) as exc:
                log.warning("invalid_persisted_suggestion", row_id=row.get("id"), error=str(exc))
                continue
            if suggestion.id in seen:
                continue
            seen.add(suggestion.id)
            self._seen_ids.add(suggestion.id)
            if suggestion.status == "expired":
                self._suggestions.pop(suggestion.id, None)
                continue
            if suggestion.status in ("accepted", "dismissed") and suggestion.response_source == "user":
                self._feedback[suggestion.id] = suggestion.status == "accepted"
            loaded += suggestion.id not in self._suggestions
            self._suggestions[suggestion.id] = suggestion

        if loaded:
            log.info("persisted_suggestions_loaded", count=loaded)
        return loaded

    def save_suggestions(self, store: MemoryStore) -> int:
        """Persist current suggestions and return the count."""
        saved = 0
        from rune.proactive.state import save

        for s in self._suggestions.values():
            save(store, s)
            saved += 1

        if saved:
            log.info("suggestions_persisted", count=saved)
        return saved

    def prune_expired_suggestions(self) -> int:
        """Remove expired suggestions and return the count."""
        now = datetime.now(UTC)
        expired_ids: list[str] = []
        for sid, s in self._suggestions.items():
            if s.expires_at and s.expires_at < now:
                expired_ids.append(sid)

        for sid in expired_ids:
            self.delete_suggestion(sid)

        if expired_ids:
            log.debug("suggestions_pruned", count=len(expired_ids))
        return len(expired_ids)

    # Public API - Stats

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        total_feedback = len(self._feedback)
        accepted_count = sum(1 for v in self._feedback.values() if v)
        acceptance_rate = (
            accepted_count / total_feedback if total_feedback > 0 else 0.0
        )
        pending_count = sum(
            1 for s in self._suggestions.values()
            if s.status == "pending" and s.execution_status is None
        )
        return {
            "evaluation_count": self._evaluation_count,
            "suggestion_count": len(self._suggestions),
            "acceptance_rate": acceptance_rate,
            "pending_count": pending_count,
            "interaction_count": total_feedback,
        }

    # Backward-compatible API

    def record_feedback(self, suggestion_id: str, accepted: bool) -> None:
        """Record user feedback on a suggestion (alias for handle_response)."""
        self.handle_response(suggestion_id, accepted)

    # Internal pipeline stages

    async def _gather_context(
        self, base_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Enrich the base context with additional signals."""
        enriched = dict(base_context)
        enriched.setdefault("timestamp", datetime.now(UTC).isoformat())
        return enriched

    async def _generate_candidates(
        self,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate candidates from hints, completed tasks, predictions and context signals."""
        candidates: list[Suggestion] = []

        # Source 1: Explicit hints
        hints: list[dict[str, Any]] = context.get("hints", [])
        for hint in hints:
            candidates.append(
                Suggestion(
                    type=hint.get("type", "insight"),
                    title=hint.get("title", ""),
                    description=hint.get("description", ""),
                    confidence=hint.get("confidence", 0.5),
                    source=hint.get("source", "context"),
                )
            )

        # Source 2: Task completion follow-up
        last_action = context.get("last_action")
        if last_action and last_action.get("status") == "completed":
            candidates.append(
                Suggestion(
                    type="followup",
                    title="Follow-up available",
                    description=f"Task '{last_action.get('goal', '')}' completed. Any follow-up?",
                    confidence=0.4,
                    source="task_completion",
                )
            )

        # Source 3: PredictionEngine (behavior + frustration + needs)
        try:
            from rune.proactive.prediction.engine import get_prediction_engine

            pred = get_prediction_engine()
            result = pred.predict(context)

            # Suggest concrete bash commands above the confidence floor, not generic tool names.
            for tool, prob in result.tool_predictions:
                if prob >= 0.6 and tool.startswith("bash:"):
                    cmd_name = tool.split(":", 1)[1]
                    from rune.utils.shell_command import command_name

                    if command_name(cmd_name) != cmd_name:
                        continue
                    candidates.append(
                        Suggestion(
                            id=f"behavior-command-{cmd_name}",
                            type="followup",
                            title=f"Run {cmd_name}?",
                            description=f"You usually run {cmd_name} here ({prob:.0%})",
                            confidence=prob,
                            source="behavior_prediction",
                        )
                    )

            # 3b: Frustration detection → warning suggestions
            if result.frustration and result.frustration.level in ("moderate", "high"):
                candidates.append(
                    Suggestion(
                        # Reuse a stable ID so heartbeats cannot duplicate the same card.
                        id=f"frustration-{result.frustration.level}",
                        type="warning",
                        title="Difficulty detected",
                        description=result.frustration.suggested_action,
                        confidence=0.6 if result.frustration.level == "moderate" else 0.75,
                        source="frustration_detection",
                    )
                )

            # 3c: Need inference → reminder suggestions
            for need in result.needs:
                if need.confidence >= 0.5:
                    candidates.append(
                        Suggestion(
                            id=f"need-{need.need_type}",
                            type="reminder",
                            title=f"Consider {need.need_type}",
                            description=_NEED_DESCRIPTIONS.get(
                                need.need_type,
                                f"Your workflow suggests {need.need_type} may be needed.",
                            ),
                            confidence=need.confidence * 0.85,
                            source="need_inference",
                        )
                    )

        except Exception as exc:
            # Prediction failure must never break the pipeline
            log.debug("prediction_engine_skipped", error=str(exc)[:200])

        # Source 4: Context-based triggers (git, idle, commitments)
        try:
            # 4a: Git dirty - uncommitted changes after idle
            git_status = context.get("git_status", "")
            if git_status.strip():
                dirty_count = len([
                    l for l in git_status.strip().splitlines() if l.strip()
                ])
                if dirty_count >= 2:
                    candidates.append(
                        Suggestion(
                            id="git-uncommitted",
                            type="reminder",
                            title="Uncommitted changes",
                            description=f"{dirty_count} files have uncommitted changes. Want to commit?",
                            confidence=0.55,
                            source="git_context",
                        )
                    )

            # 4b: Idle detection - user may be stuck
            idle_secs = context.get("idle_seconds", 0)
            if idle_secs and idle_secs >= 180:
                candidates.append(
                    Suggestion(
                        id="idle-detected",
                        type="insight",
                        title="Idle detected",
                        description="You seem idle. Need help with anything?",
                        confidence=0.45,
                        source="idle_detection",
                    )
                )

            # 4c: Open commitments from episode memory
            store = None
            try:
                from rune.memory.manager import get_memory_manager
                mgr = get_memory_manager()
                store = getattr(mgr, "store", None)
            except Exception as exc:
                log.debug("open_commitments_store_unavailable", error=str(exc)[:100])

            if store is not None and hasattr(store, "get_open_commitments"):
                open_commits = store.get_open_commitments(limit=2)
                for c in open_commits:
                    candidates.append(
                        Suggestion(
                            # Deduplicate heartbeats and UI events by commitment ID.
                            id=f"commitment-{c['id']}",
                            type="followup",
                            title="Open commitment",
                            description=f"Pending: {c['text'][:100]}",
                            confidence=0.6,
                            source="commitment_tracking",
                        )
                    )

        except Exception as exc:
            log.debug("context_triggers_skipped", error=str(exc)[:200])

        return candidates

    def _filter_candidates(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Filter expiry and confidence using a stricter learned threshold when available."""
        now = datetime.now(UTC)
        min_confidence = self._config.get("min_confidence", 0.2)

        # Apply reflexion-learned threshold override (may be higher than default)
        try:
            from rune.proactive.reflexion import get_reflexion_learner
            learned_threshold = get_reflexion_learner().get_score_threshold()
            if learned_threshold is not None and learned_threshold > min_confidence:
                min_confidence = learned_threshold
        except Exception as exc:
            log.debug("reflexion_threshold_unavailable", error=str(exc)[:100])

        filtered: list[Suggestion] = []
        for s in candidates:
            if s.expires_at and s.expires_at < now:
                continue
            if s.confidence < min_confidence:
                continue
            filtered.append(s)
        return filtered

    def _rank_candidates(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Use explicit feedback to adjust relevance, never execution authority."""
        by_type: dict[str, list[bool]] = {}
        for sid, accepted in self._feedback.items():
            if previous := self._suggestions.get(sid):
                by_type.setdefault(previous.type, []).append(accepted)

        def score(s: Suggestion) -> float:
            feedback = by_type.get(s.type, [])
            adjustment = (sum(feedback) / len(feedback) - .5) * .2 if len(feedback) >= 3 else 0
            return s.confidence + adjustment

        return sorted(candidates, key=score, reverse=True)

    def _deduplicate(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Remove suggestions with duplicate titles or already-seen IDs."""
        seen_titles: set[str] = set()
        result: list[Suggestion] = []
        for s in candidates:
            title_key = s.title.lower().strip()
            if s.id in self._seen_ids or s.id in self._suggestions:
                continue
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            result.append(s)
        return result


def get_proactive_engine(config: dict[str, Any] | None = None) -> ProactiveEngine:
    """Get or create the singleton ProactiveEngine."""
    global _engine
    if _engine is None:
        _engine = ProactiveEngine(config)
    return _engine
