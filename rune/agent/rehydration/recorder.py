"""Compaction recorder — markdown-first storage for compacted messages.

Writes to ``~/.rune/memory/compacted/`` (source of truth) and then
updates the FAISS index (derived cache).  If FAISS fails, the markdown
record is preserved and the index can be rebuilt later.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from rune.agent.rehydration.protocols import _MAX_PENDING, CompactedRecord
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Helpers
_RECORD_RE = re.compile(
    r"^## (?P<ts>[^\|]+)\| step=(?P<step>\d+) \| "
    r"phase=(?P<phase>\w+) \| role=(?P<role>\w+)"
    r"(?:\s*\| tool=(?P<tool>\S+))?",
)


def _compacted_dir() -> Path:
    from rune.utils.paths import rune_home

    return rune_home() / "memory" / "compacted"


def _extract_tool_name(msg: dict[str, Any]) -> str:
    """Extract tool name from a message dict, if present."""
    # tool-result messages
    if msg.get("role") == "tool":
        return msg.get("name", "")
    # assistant messages with tool_calls
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        return fn.get("name", "")
    return ""


def _render_message(msg: dict[str, Any]) -> str:
    """Render a message dict to a string for storage."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    return str(content)[:4000]


# Compaction Recorder
class CompactionRecorder:
    """Append-only recorder for compacted messages.

    Writes markdown first (source of truth), then updates FAISS (derived).
    Both writes are async; markdown uses fcntl-based atomic append;
    FAISS uses existing VectorStore.add_batch().
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._md_path = self._resolve_path()
        self._pending: list[CompactedRecord] = []
        self._flush_lock = asyncio.Lock()

    def _resolve_path(self) -> Path:
        from datetime import UTC, datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return _compacted_dir() / f"{today}_session-{self._session_id[:12]}.md"

    async def record(
        self,
        messages: list[dict[str, Any]],
        *,
        step_range: tuple[int, int],
        activity_phase: str,
        compaction_event: str,
    ) -> None:
        """Queue messages for recording. Triggers async flush."""
        for i, msg in enumerate(messages):
            self._pending.append(
                CompactedRecord(
                    step=step_range[0] + i,
                    activity_phase=activity_phase,
                    role=msg.get("role", "unknown"),
                    tool_name=_extract_tool_name(msg),
                    original_content=_render_message(msg),
                    compaction_event=compaction_event,
                )
            )
        # Fire-and-forget flush — exception handled inside
        asyncio.create_task(self._flush(), name="rehydration_flush")

    async def flush(self) -> None:
        """Public flush for shutdown."""
        await self._flush()

    async def _flush(self) -> None:
        async with self._flush_lock:
            if not self._pending:
                return
            batch = self._pending[:]
            self._pending.clear()

        try:
            # Step 1: Append to markdown (source of truth)
            from rune.memory.markdown_store import _atomic_append

            md_text = "\n\n".join(r.to_markdown() for r in batch)
            _atomic_append(self._md_path, md_text + "\n\n")
            log.debug("compacted_md_written", count=len(batch), path=str(self._md_path))

            # Step 2: Update FAISS (derived cache, best-effort)
            try:
                from rune.memory.manager import get_memory_manager

                mgr = get_memory_manager()
                texts = [r.original_content for r in batch]
                embeddings = await mgr.embed_batch(texts)
                from rune.memory.types import VectorMetadata

                for emb, record in zip(embeddings, batch, strict=False):
                    meta_dict = record.to_vector_metadata(self._session_id)
                    meta = VectorMetadata(
                        type=meta_dict["type"],
                        id=meta_dict["id"],
                        timestamp=meta_dict["timestamp"],
                        summary=meta_dict["summary"],
                        category=meta_dict["category"],
                        extra=meta_dict["extra"],
                    )
                    mgr._vectors.add(emb, meta)
                log.debug("compacted_faiss_indexed", count=len(batch))
            except Exception as exc:
                log.warning("compacted_faiss_failed", error=str(exc)[:200])
                # Markdown has the records — FAISS can be rebuilt later

        except Exception as exc:
            log.warning("compacted_record_failed", error=str(exc)[:200])
            # Re-queue with soft cap to prevent unbounded growth
            requeue = batch + self._pending
            self._pending = requeue[:_MAX_PENDING]
            if len(requeue) > _MAX_PENDING:
                log.warning(
                    "compacted_pending_capped",
                    dropped=len(requeue) - _MAX_PENDING,
                )

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# Rebuild — reconstruct FAISS from markdown source of truth
def parse_compacted_md(path: Path) -> list[CompactedRecord]:
    """Parse a compacted markdown file into records."""
    if not path.exists():
        return []

    records: list[CompactedRecord] = []
    text = path.read_text(encoding="utf-8")
    sections = text.split("\n## ")

    for section in sections:
        if not section.strip():
            continue
        # Re-add the ## prefix stripped by split
        header_line = section.split("\n", 1)[0]
        m = _RECORD_RE.match("## " + header_line)
        if not m:
            continue
        body = section.split("\n", 1)[1] if "\n" in section else ""
        # Extract original content between markers
        content = ""
        if "**Original content**" in body:
            start = body.index("**Original content**")
            rest = body[start:]
            lines = rest.split("\n")[1:]  # skip the header line
            content_lines = []
            for line in lines:
                if line.startswith("**Compaction event**"):
                    break
                content_lines.append(line)
            content = "\n".join(content_lines).strip()

        records.append(
            CompactedRecord(
                step=int(m.group("step")),
                activity_phase=m.group("phase"),
                role=m.group("role"),
                tool_name=m.group("tool") or "",
                original_content=content,
                compaction_event="rebuilt",
                timestamp=m.group("ts").strip(),
            )
        )
    return records


async def rebuild_compacted_index(session_id: str) -> int:
    """Rebuild the FAISS index from the markdown log.

    Returns the number of records indexed.
    """
    md_files = list(_compacted_dir().glob(f"*_session-{session_id[:12]}.md"))
    if not md_files:
        return 0

    all_records: list[CompactedRecord] = []
    for md_path in md_files:
        all_records.extend(parse_compacted_md(md_path))

    if not all_records:
        return 0

    from rune.memory.manager import get_memory_manager
    from rune.memory.types import VectorMetadata

    mgr = get_memory_manager()
    texts = [r.original_content for r in all_records]
    embeddings = await mgr.embed_batch(texts)

    for emb, record in zip(embeddings, all_records, strict=False):
        meta_dict = record.to_vector_metadata(session_id)
        meta = VectorMetadata(
            type=meta_dict["type"],
            id=meta_dict["id"],
            timestamp=meta_dict["timestamp"],
            summary=meta_dict["summary"],
            category=meta_dict["category"],
            extra=meta_dict["extra"],
        )
        mgr._vectors.add(emb, meta)

    log.info("compacted_index_rebuilt", session_id=session_id, count=len(all_records))
    return len(all_records)
