"""Proactive suggestion engine for RUNE.

Evaluates the current context and produces ranked, deduplicated suggestions
through an 8-step pipeline.  Also provides full CRUD, persistence, and
deduplication-cooldown for suggestions.
"""

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
    """Generates proactive suggestions via an 8-step evaluation pipeline.

    Also provides:
    - In-memory suggestion storage with CRUD
    - Persistence via MemoryStore's proactive_suggestions_state table
    - Deduplication cooldown (5-min per title)
    - Stats reporting
    - Event emission for suggestion/intervention/decision listeners
    """

    __slots__ = (
        "_config",
        "_feedback",
        "_seen_ids",
        "_suggestions",
        "_recent_suggestion_keys",
        "_evaluation_count",
        "_listeners",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._feedback: dict[str, bool] = {}  # suggestion_id -> accepted
        self._seen_ids: set[str] = set()
        self._suggestions: dict[str, Suggestion] = {}  # id -> Suggestion
        self._recent_suggestion_keys: dict[str, datetime] = {}  # title_key -> last_added
        self._evaluation_count: int = 0
        self._listeners: dict[str, list[Any]] = {}  # event_name -> [callbacks]

    # Event emitter (ported from TS EventEmitter pattern)

    def on(self, event: str, callback: Any) -> None:
        """Register a listener for an event.

        Events:
        - ``suggestion``: emitted when new suggestions are produced
        - ``intervention``: emitted when an intervention is triggered
        - ``decision``: emitted when the engine makes a decision
        - ``task_completed``: emitted when a task completes
        - ``task_failed``: emitted when a task fails
        """
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
                log.warning("event_listener_error", event=event, error=str(exc))

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
        """Run the 8-step suggestion pipeline.

        Steps:
        1. Gather context
        2. Check preconditions (quiet hours, suppression)
        3. Generate candidates
        4. Filter candidates (relevance, expiry)
        5. Rank candidates (confidence, user preference)
        6. Deduplicate
        7. Limit output count
        8. Record pipeline metadata
        """
        self._evaluation_count += 1

        # Step 1: Gather context
        enriched = await self._gather_context(context)

        # Step 2: Preconditions
        if enriched.get("suppress", False):
            return []

        # Step 3: Generate
        candidates = await self._generate_candidates(enriched)

        # Step 4: Filter
        candidates = self._filter_candidates(candidates)

        # Step 5: Rank
        candidates = self._rank_candidates(candidates)

        # Step 6: Deduplicate
        candidates = self._deduplicate(candidates)

        # Step 7: Limit
        max_suggestions = self._config.get("max_suggestions", 3)
        candidates = candidates[:max_suggestions]

        # Step 8: Record (with eviction to bound memory) + store in _suggestions
        for s in candidates:
            self._seen_ids.add(s.id)
            self._suggestions[s.id] = s
        if len(self._seen_ids) > _MAX_SEEN_IDS:
            # Evict oldest half - set iteration order is insertion order in CPython 3.7+
            to_remove = list(self._seen_ids)[: _MAX_SEEN_IDS // 2]
            for item in to_remove:
                self._seen_ids.discard(item)

        log.debug("proactive_evaluated", count=len(candidates))

        # Emit events for new suggestions
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
        """Manually inject a suggestion into the engine.

        Respects deduplication cooldown: if the same title was added within
        the last 5 minutes the suggestion is silently dropped.
        """
        # Dedup cooldown check
        title_key = suggestion.title.lower().strip()
        now = datetime.now(UTC)

        if title_key and title_key in self._recent_suggestion_keys:
            last_added = self._recent_suggestion_keys[title_key]
            if (now - last_added).total_seconds() < _DEDUP_COOLDOWN_SECS:
                log.debug("suggestion_dedup_cooldown", title=suggestion.title)
                return

        self._recent_suggestion_keys[title_key] = now
        self._suggestions[suggestion.id] = suggestion

        # Prune stale cooldown entries (keep at most 200)
        if len(self._recent_suggestion_keys) > 200:
            cutoff = now - timedelta(seconds=_DEDUP_COOLDOWN_SECS)
            self._recent_suggestion_keys = {
                k: v
                for k, v in self._recent_suggestion_keys.items()
                if v > cutoff
            }

        log.debug("suggestion_added", id=suggestion.id)

    def get_suggestion(self, suggestion_id: str) -> Suggestion | None:
        """Fetch a suggestion by ID."""
        return self._suggestions.get(suggestion_id)

    def get_first_pending(self) -> Suggestion | None:
        """Return the oldest unprocessed suggestion above min_confidence.

        Returns ``None`` if no qualifying suggestion exists.
        """
        min_confidence = self._config.get("min_confidence", 0.2)
        now = datetime.now(UTC)

        # Iterate in insertion order (oldest first in CPython 3.7+)
        for s in self._suggestions.values():
            if s.status != "pending":
                continue
            if s.confidence < min_confidence:
                continue
            if s.expires_at and s.expires_at < now:
                continue
            return s
        return None

    def delete_suggestion(self, suggestion_id: str) -> None:
        """Remove a suggestion by ID."""
        self._suggestions.pop(suggestion_id, None)

    def handle_response(self, suggestion_id: str, accepted: bool) -> None:
        """Process user feedback on a suggestion.

        Updates the feedback dict and removes the suggestion from pending.
        """
        self._feedback[suggestion_id] = accepted

        suggestion = self._suggestions.get(suggestion_id)
        if suggestion is not None:
            suggestion.status = "accepted" if accepted else "dismissed"
            # Remove from pending storage
            del self._suggestions[suggestion_id]

        log.debug(
            "proactive_feedback",
            suggestion_id=suggestion_id,
            accepted=accepted,
        )

    # Public API - Persistence

    def load_persisted_suggestions(self, store: MemoryStore) -> int:
        """Load suggestions from the store's proactive_suggestions_state table.

        Returns the number of suggestions loaded.
        """
        rows = store.get_suggestion_state()  # returns list[dict]
        loaded = 0
        for row in rows:
            sid = str(row.get("id", ""))
            meta: dict[str, Any] = row.get("metadata", {})
            if sid in self._suggestions:
                continue

            suggestion = Suggestion(
                id=meta.get("suggestion_id", sid),
                type=meta.get("type", "insight"),
                title=meta.get("title", ""),
                description=meta.get("description", ""),
                confidence=float(meta.get("confidence", 0.5)),
                source=meta.get("source", "persisted"),
                status=row.get("state", "pending"),
            )

            # Restore timestamps if present
            if meta.get("created_at"):
                with contextlib.suppress(ValueError, TypeError):
                    suggestion.created_at = datetime.fromisoformat(meta["created_at"])
            if meta.get("expires_at"):
                with contextlib.suppress(ValueError, TypeError):
                    suggestion.expires_at = datetime.fromisoformat(meta["expires_at"])

            self._suggestions[suggestion.id] = suggestion
            loaded += 1

        if loaded:
            log.info("persisted_suggestions_loaded", count=loaded)
        return loaded

    def save_suggestions(self, store: MemoryStore) -> int:
        """Persist current in-memory suggestions to the store.

        Returns the number of suggestions saved.
        """
        saved = 0
        for s in self._suggestions.values():
            meta = {
                "suggestion_id": s.id,
                "type": s.type,
                "title": s.title,
                "description": s.description,
                "confidence": s.confidence,
                "source": s.source,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
            }
            store.save_suggestion_state(
                suggestion_type=s.type,
                state=s.status,
                metadata=meta,
            )
            saved += 1

        if saved:
            log.info("suggestions_persisted", count=saved)
        return saved

    def prune_expired_suggestions(self) -> int:
        """Remove expired suggestions from in-memory storage.

        Returns the number pruned.
        """
        now = datetime.now(UTC)
        expired_ids: list[str] = []
        for sid, s in self._suggestions.items():
            if s.expires_at and s.expires_at < now:
                expired_ids.append(sid)

        for sid in expired_ids:
            self._suggestions[sid].status = "expired"
            del self._suggestions[sid]

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
            1 for s in self._suggestions.values() if s.status == "pending"
        )
        return {
            "evaluation_count": self._evaluation_count,
            "suggestion_count": len(self._suggestions),
            "acceptance_rate": acceptance_rate,
            "pending_count": pending_count,
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
        """Generate suggestion candidates from the gathered context.

        Sources:
        1. Explicit hints from context dict
        2. Task completion follow-ups
        3. PredictionEngine (behavior, frustration, needs)
        """
        candidates: list[Suggestion] = []

        # --- Source 1: Explicit hints ---
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

        # --- Source 2: Task completion follow-up ---
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

        # --- Source 3: PredictionEngine (behavior + frustration + needs) ---
        try:
            from rune.proactive.prediction.engine import get_prediction_engine

            pred = get_prediction_engine()
            result = pred.predict(context)

            # 3a: Tool behavior predictions - used internally for
            # agent optimization, NOT surfaced as user-facing suggestions.
            # The agent already generates contextual follow-up suggestions
            # in its responses; tool-level predictions would be redundant
            # noise (e.g., "web_search is likely useful next" is obvious
            # and adds no value to the user).
            # Tool predictions are still available via result.tool_predictions
            # for internal use (e.g., pre-warming, tool prioritization).

            # 3b: Frustration detection → warning suggestions
            if result.frustration and result.frustration.level in ("moderate", "high"):
                candidates.append(
                    Suggestion(
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

        # --- Source 4: Context-based triggers (git, idle, commitments) ---
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
                            type="reminder",
                            title="Uncommitted changes",
                            description=f"커밋하지 않은 변경이 {dirty_count}개 파일에 있어요. 커밋할까요?",
                            confidence=0.55,
                            source="git_context",
                        )
                    )

            # 4b: Idle detection - user may be stuck
            idle_secs = context.get("idle_seconds", 0)
            if idle_secs and idle_secs >= 180:
                candidates.append(
                    Suggestion(
                        type="insight",
                        title="Idle detected",
                        description="잠시 멈춰있는 것 같아요. 뭔가 도와드릴까요?",
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
            except Exception:
                pass

            if store is not None and hasattr(store, "get_open_commitments"):
                open_commits = store.get_open_commitments(limit=2)
                for c in open_commits:
                    candidates.append(
                        Suggestion(
                            type="followup",
                            title="Open commitment",
                            description=f"미완료 항목: {c['text'][:100]}",
                            confidence=0.6,
                            source="commitment_tracking",
                        )
                    )

        except Exception as exc:
            log.debug("context_triggers_skipped", error=str(exc)[:200])

        return candidates

    def _filter_candidates(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Remove expired or low-confidence candidates.

        Uses reflexion-learned threshold if available (higher than default
        when users have been rejecting suggestions).
        """
        now = datetime.now(UTC)
        min_confidence = self._config.get("min_confidence", 0.2)

        # Apply reflexion-learned threshold override (may be higher than default)
        try:
            from rune.proactive.reflexion import get_reflexion_learner
            learned_threshold = get_reflexion_learner().get_score_threshold()
            if learned_threshold is not None and learned_threshold > min_confidence:
                min_confidence = learned_threshold
        except Exception:
            pass

        filtered: list[Suggestion] = []
        for s in candidates:
            if s.expires_at and s.expires_at < now:
                continue
            if s.confidence < min_confidence:
                continue
            filtered.append(s)
        return filtered

    def _rank_candidates(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Rank candidates by confidence (desc), boosted by positive feedback history."""
        acceptance_count = sum(1 for v in self._feedback.values() if v)
        total_feedback = len(self._feedback)
        global_boost = (acceptance_count / max(1, total_feedback)) * 0.1

        def score(s: Suggestion) -> float:
            return s.confidence + global_boost

        return sorted(candidates, key=score, reverse=True)

    def _deduplicate(self, candidates: list[Suggestion]) -> list[Suggestion]:
        """Remove suggestions with duplicate titles or already-seen IDs."""
        seen_titles: set[str] = set()
        result: list[Suggestion] = []
        for s in candidates:
            title_key = s.title.lower().strip()
            if s.id in self._seen_ids:
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
