"""Episode consolidation for RUNE.

Background job that sends recent episodes to a fast LLM to extract
structured metadata: commitments, lessons, entities, decisions.

Replaces regex-based extraction with LLM understanding.
Runs asynchronously - zero impact on real-time agent performance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Configuration

_CONSOLIDATION_ENABLED_KEY = "learning"  # rune config key
_DEFAULT_ENABLED = True
_MAX_EPISODES_PER_RUN = 5  # process at most N episodes per tick
_EXTRACTION_MAX_TOKENS = 512


# Extraction prompt

_EXTRACTION_PROMPT = """\
You are a structured data extractor. Given a task summary and result from \
an AI assistant session, extract the following in JSON format:

{
  "commitments": ["things the user said they need to do, with deadlines if mentioned"],
  "lessons": ["reusable insights - what worked, what failed, how it was fixed"],
  "entities": ["people, projects, services, files mentioned"],
  "decisions": ["key choices or conclusions made"]
}

Rules:
- Only extract what is EXPLICITLY stated. Do not infer or guess.
- Commitments must be user obligations, not completed tasks.
- Lessons must be actionable for future reference.
- Entities are proper nouns: names, project names, file paths, service names.
- Return empty arrays if nothing fits a category.
- Respond with ONLY the JSON object, no other text.
"""


# Consolidation result

def _parse_extraction(raw: str) -> dict[str, list[str]]:
    """Parse LLM extraction output into structured dict."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                return {"commitments": [], "lessons": [], "entities": [], "decisions": []}
        else:
            return {"commitments": [], "lessons": [], "entities": [], "decisions": []}

    return {
        "commitments": data.get("commitments", [])[:10],
        "lessons": data.get("lessons", [])[:10],
        "entities": data.get("entities", [])[:20],
        "decisions": data.get("decisions", [])[:10],
    }


# Public API

_LEARNING_SETTING_FILE = "learning_enabled"


def _setting_path() -> str:
    """Return path to the learning setting file."""
    from rune.utils.paths import rune_data
    return str(rune_data() / _LEARNING_SETTING_FILE)


def is_consolidation_enabled() -> bool:
    """Check if background learning is enabled.

    Priority: env var > setting file > default (None = not yet decided).
    """
    import os

    # 1. Environment variable override
    env = os.environ.get("RUNE_LEARNING", "").lower()
    if env in ("1", "true", "on", "yes"):
        return True
    if env in ("0", "false", "off", "no"):
        return False

    # 2. Persisted setting file
    try:
        path = _setting_path()
        if os.path.exists(path):
            with open(path) as f:
                val = f.read().strip().lower()
            return val in ("1", "true", "on", "yes")
    except Exception:
        pass

    # 3. Not yet decided - default enabled but first-run notice needed
    return _DEFAULT_ENABLED


def is_first_run() -> bool:
    """Check if learning has never been configured (no setting file exists)."""
    import os
    try:
        return not os.path.exists(_setting_path())
    except Exception:
        return True


def set_learning_enabled(enabled: bool) -> None:
    """Persist the learning on/off setting."""
    try:
        path = _setting_path()
        with open(path, "w") as f:
            f.write("on" if enabled else "off")
    except Exception as exc:
        log.warning("learning_setting_save_failed", error=str(exc)[:100])


async def consolidate_episode(episode_id: str) -> dict[str, list[str]] | None:
    """Extract structured metadata from a single episode using fast LLM.

    Returns the extraction result dict, or None if extraction failed
    or consolidation is disabled.
    """
    if not is_consolidation_enabled():
        return None

    try:
        from rune.llm.client import get_llm_client
        from rune.memory.manager import get_memory_manager

        mgr = get_memory_manager()
        store = getattr(mgr, "store", None)
        if store is None:
            return None

        # Fetch the episode
        episodes = store.get_recent_episodes(limit=50)
        episode = next((e for e in episodes if e.id == episode_id), None)
        if episode is None:
            return None

        # Skip if already consolidated (entities field is populated)
        if episode.entities and episode.entities.strip():
            return None

        # Build extraction input
        input_text = f"Task: {episode.task_summary}\nResult: {episode.result}"
        if episode.lessons:
            input_text += f"\nLessons: {episode.lessons}"

        # Call fast model
        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": input_text},
            ],
            tier="fast",
            temperature=0.0,
            max_tokens=_EXTRACTION_MAX_TOKENS,
            timeout=15.0,
        )

        # Extract text from response
        response_text = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
        else:
            with contextlib.suppress(AttributeError, IndexError):
                response_text = response.choices[0].message.content

        if not response_text:
            return None

        result = _parse_extraction(response_text)

        # Save back to episode
        import json as _json
        store.conn.execute(
            """UPDATE episodes SET
                entities = ?,
                commitments = ?,
                lessons = CASE WHEN lessons = '' OR lessons IS NULL THEN ? ELSE lessons END
               WHERE id = ?""",
            (
                _json.dumps(result["entities"]) if result["entities"] else "",
                _json.dumps(result["commitments"]) if result["commitments"] else "",
                "; ".join(result["lessons"]) if result["lessons"] else "",
                episode_id,
            ),
        )

        # Save commitments to dedicated table
        if result["commitments"] and hasattr(store, "save_commitment"):
            for c in result["commitments"]:
                if isinstance(c, str) and len(c) > 5:
                    store.save_commitment(episode_id, c)

        # Save decisions and lessons to learned.md
        try:
            from rune.memory.markdown_store import save_learned_fact
            from rune.memory.state import is_suppressed

            for d in result.get("decisions", []):
                if isinstance(d, str) and len(d) > 5 and not is_suppressed(d[:80]):
                    save_learned_fact("decision", d[:80], d, 0.8)

            for lesson in result.get("lessons", []):
                if isinstance(lesson, str) and len(lesson) > 5 and not is_suppressed(lesson[:40]):
                    save_learned_fact("lesson", lesson[:40], lesson, 0.7)
        except Exception as exc:
            log.debug("learned_md_write_failed", error=str(exc)[:100])

        log.debug(
            "episode_consolidated",
            episode_id=episode_id,
            commitments=len(result["commitments"]),
            lessons=len(result["lessons"]),
            entities=len(result["entities"]),
            decisions=len(result["decisions"]),
        )

        return result

    except Exception as exc:
        log.debug("consolidation_failed", episode_id=episode_id, error=str(exc)[:200])
        return None


async def consolidate_recent(limit: int = _MAX_EPISODES_PER_RUN) -> int:
    """Consolidate recent unconsolidated episodes. Returns count processed."""
    if not is_consolidation_enabled():
        return 0

    try:
        from rune.memory.manager import get_memory_manager

        mgr = get_memory_manager()
        store = getattr(mgr, "store", None)
        if store is None:
            return 0

        # Find episodes without entities (not yet consolidated)
        episodes = store.get_recent_episodes(limit=50)
        unconsolidated = [
            e for e in episodes
            if not (e.entities and e.entities.strip())
            and e.task_summary  # skip empty episodes
        ][:limit]

        if not unconsolidated:
            return 0

        count = 0
        for ep in unconsolidated:
            result = await consolidate_episode(ep.id)
            if result is not None:
                count += 1
            # Small delay between calls to avoid rate limiting
            await asyncio.sleep(0.5)

        if count:
            log.info("consolidation_batch_complete", processed=count)

        return count

    except Exception as exc:
        log.debug("consolidation_batch_failed", error=str(exc)[:200])
        return 0
