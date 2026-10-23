"""One receipt per external change, graded by the evidence behind it.

Tiers, strongest first:
  verified      the change was read back independently and matched (file hash, absence, stored record)
  acknowledged  the receiver accepted it (HTTP 2xx, tool success) but nothing re-read the stored result
  observed      the change was seen on disk after a shell command, without a request to compare it with
  rejected      the receiver refused it before acting (4xx, 503); nothing should have changed
  unknown       the request may have reached the receiver but no outcome was observed

File receipts are keyed by path, so a repaired write replaces its earlier failure. Every other
change is keyed by its tool call: two messages to the same channel are two changes, and a later
failure must not hide an earlier success.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

_MAX_HASH_BYTES = 16 * 1024 * 1024
_MAX_RECEIPTS = 64
_MAX_PER_KIND = 20
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Tool caches a command regenerates as a side effect; listing them would bury the real changes.
_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_STATE_TOOLS = {"cron_create": "name", "cron_delete": "name", "cron_update": "job_id",
                "credential_save": "key", "skill_create": "name"}
_DOCUMENT_TOOLS = frozenset({"document_create", "document_bundle", "document_bundle_update"})


def redact_url(url: Any) -> str:
    """Drop credentials, query and fragment: tokens often travel there and receipts are shown and stored."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path, "…" if parts.query else "", ""))
    except ValueError:
        return "(unparsed URL)"


def _sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_HASH_BYTES:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        log.debug("receipt_unreadable", path=str(path), error=str(exc))
        return None


def _http_tier(status: Any) -> str:
    if not isinstance(status, int) or status <= 0 or status == 202:
        return "unknown"
    if 200 <= status < 300:
        return "acknowledged"
    # A 500/502/504 can follow a committed write; only a refusal shows nothing changed.
    return "rejected" if 400 <= status < 500 or status == 503 else "unknown"


def _http_receipt(method: str, target: str, meta: dict) -> dict:
    status = meta.get("status_code")
    return {"target": f"{method} {target}", "tier": _http_tier(status),
            "evidence": f"HTTP {status}" if isinstance(status, int) and status else "No response status was observed."}


def _file_receipt(tool: str, params: dict, result: CapabilityResult, expected_edit: str | None) -> dict | None:
    meta = result.metadata or {}
    raw = meta.get("path") or params.get("path") or params.get("file_path")
    if not result.success or not isinstance(raw, str) or meta.get("changed") is False:
        return None
    path = Path(raw).expanduser()
    if tool == "file_delete":
        try:
            path.lstat()
            return {"tier": "unknown", "evidence": "The path still exists after the delete."}
        except FileNotFoundError:
            return {"tier": "verified", "evidence": "The path is absent after the delete."}
        except OSError as exc:
            log.debug("receipt_absence_unconfirmed", path=str(path), error=str(exc))
            return {"tier": "unknown", "evidence": "The path could not be checked after the delete."}
    actual = _sha256(path)
    expected = None
    if tool == "file_write" and isinstance(params.get("content"), str):
        try:
            expected = hashlib.sha256(params["content"].encode(params.get("encoding") or "utf-8")).hexdigest()
        except (LookupError, UnicodeError) as exc:
            log.debug("receipt_encoding_failed", error=str(exc))
    elif tool == "file_edit":
        expected = expected_edit
    if actual is not None and expected is not None:
        return ({"tier": "verified", "evidence": "The file was read back and its content hash matches the change."}
                if actual == expected else
                {"tier": "unknown", "evidence": "The file was read back but its content differs from the requested change."})
    return {"tier": "acknowledged", "evidence": "The tool reported the change; its exact content was not compared."}


def may_change(tool: str, params: dict[str, Any]) -> bool:
    """Whether a call can change something outside the conversation, before its result is known."""
    method = str(params.get("method") or "GET").upper()
    if tool in {"file_write", "file_edit", "file_delete", "browser_act", "browser_batch", "browser_workflow"}:
        return True
    if tool in _DOCUMENT_TOOLS or tool in _STATE_TOOLS:
        return True
    if tool in {"web_fetch", "connector_request"}:
        return method not in _SAFE_METHODS
    if tool == "bash_execute":
        from rune.safety.workspace_snapshot import looks_read_only

        return not looks_read_only(str(params.get("command") or ""))
    if tool.startswith("mcp."):
        from rune.safety.tool_policy import mcp_changes_state

        return mcp_changes_state(tool)
    return False


def _pending_target(tool: str, params: dict[str, Any]) -> str:
    if tool.startswith("file_") or tool in _DOCUMENT_TOOLS:
        return str(params.get("path") or params.get("file_path") or params.get("directory") or tool)
    if tool in {"web_fetch", "connector_request"}:
        where = redact_url(params.get("url")) if tool == "web_fetch" else f"{params.get('origin', '')}{params.get('path', '')}"
        return f"{str(params.get('method') or 'GET').upper()} {where}"
    if tool == "bash_execute":
        return f"shell in {params.get('cwd') or '.'}"
    if tool in _STATE_TOOLS:
        return f"{tool} {params.get(_STATE_TOOLS[tool]) or ''}".strip()
    return tool


def receipt_for(tool: str, params: dict[str, Any], result: CapabilityResult, *,
                expected_edit: str | None = None) -> dict[str, Any] | None:
    """Grade one tool result, or None when it made no external change."""
    meta = result.metadata or {}
    if meta.get("action_status") == "not_executed" or meta.get("requires_approval"):
        return None
    if tool in {"file_write", "file_edit", "file_delete"}:
        graded = _file_receipt(tool, params, result, expected_edit)
        target = meta.get("path") or params.get("path") or params.get("file_path") or ""
        return None if graded is None else {"tool": tool, "target": str(target)[:240], **graded}
    if tool == "web_fetch" and str(params.get("method") or "GET").upper() not in _SAFE_METHODS:
        return {"tool": tool, **_http_receipt(str(params["method"]).upper(), redact_url(params.get("url", "")), meta)}
    if tool == "connector_request" and str(params.get("method") or "GET").upper() not in _SAFE_METHODS:
        where = f"{params.get('connector', '')}: {params.get('path', '')}"
        return {"tool": tool, **_http_receipt(str(params["method"]).upper(), where[:200], meta)}
    if tool.startswith("mcp."):
        from rune.safety.tool_policy import mcp_changes_state

        if not mcp_changes_state(tool, read_only_hint=meta.get("mcp_read_only")):
            return None
        graded = ({"tier": "acknowledged", "evidence": "The service accepted the call; its stored result was not read back."}
                  if result.success else
                  # A tool-level error can arrive after partial side effects, so it does not prove nothing changed.
                  {"tier": "unknown", "evidence": "The service reported an error; part of the call may have applied."}
                  if meta.get("mcp_error") else
                  {"tier": "unknown", "evidence": "The call may have reached the service, but no response was received."})
        return {"tool": tool, "target": tool, **graded}
    if tool in _DOCUMENT_TOOLS and result.success:
        paths = meta.get("paths") or [meta.get("path")]
        target = ", ".join(str(path) for path in paths if path)[:240]
        if meta.get("verified"):
            return {"tool": tool, "target": target, "tier": "verified", "evidence": "The saved document passed its content checks."}
        return {"tool": tool, "target": target, "tier": "acknowledged", "evidence": "The document was written; its content was not re-read."}
    if tool in _STATE_TOOLS and result.success:
        return {"tool": tool, "target": _pending_target(tool, params), "tier": "acknowledged",
                "evidence": "Rune recorded the change; it takes effect when next used."}
    if tool == "browser_act" and isinstance(meta.get("write_receipt"), dict):
        receipt = meta["write_receipt"]
        requests = receipt.get("requests") or []
        tier = {"accepted": "acknowledged", "rejected": "rejected"}.get(receipt.get("status"), "unknown")
        return {"tool": tool, "target": ", ".join(f"{r.get('method')} {redact_url(r.get('url'))}" for r in requests[:2])[:240],
                "tier": tier, "seqs": list(receipt.get("seqs") or []),
                "evidence": ", ".join(f"HTTP {r.get('status')}" for r in requests[:4]) or "No response status was observed."}
    return None


class ReceiptLedger:
    """Ordered receipts for one run, with placeholders for calls that have not finished."""

    def __init__(self) -> None:
        self._receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self._omitted = 0
        # Removed path -> (workspace-relative path, pre-removal copy or "").
        self._removed: dict[str, tuple[str, str]] = {}

    def _put(self, key: tuple[str, str], receipt: dict[str, Any]) -> None:
        self._receipts.pop(key, None)
        if len(self._receipts) >= _MAX_RECEIPTS:
            # Shell observations are the most numerous and least specific; drop those first.
            victim = next((k for k, r in self._receipts.items() if r["tier"] == "observed"), next(iter(self._receipts)))
            self._receipts.pop(victim)
            self._omitted += 1
        self._receipts[key] = receipt

    def begin(self, call_id: str, tool: str, params: dict[str, Any]) -> None:
        """Hold an unknown receipt until the call reports, so a stop mid-call still lists it."""
        try:
            if call_id and may_change(tool, params):
                self._put(("call", call_id), {"tool": tool, "target": _pending_target(tool, params)[:240], "tier": "unknown",
                                              "evidence": "The call started but no result was recorded before the run stopped."})
        except Exception as exc:  # Receipts describe evidence; they must never break a run.
            log.debug("receipt_begin_failed", tool=tool, error=str(exc))

    def discard(self, call_id: str) -> None:
        self._receipts.pop(("call", call_id), None)

    def record(self, tool: str, params: dict[str, Any], result: CapabilityResult, *,
               expected_edit: str | None = None, call_id: str = "") -> None:
        try:
            self._receipts.pop(("call", call_id), None)
            self._record(tool, params, result, expected_edit, call_id)
        except Exception as exc:  # Receipts describe evidence; they must never break a run.
            log.debug("receipt_failed", tool=tool, error=str(exc))

    def _record(self, tool: str, params: dict[str, Any], result: CapabilityResult,
                expected_edit: str | None, call_id: str) -> None:
        if tool == "bash_execute":
            self._record_shell(params, result, call_id)
            return
        if tool in {"browser_batch", "browser_workflow"}:
            for index, step in enumerate((result.metadata or {}).get("browser_steps", [])):
                if isinstance(step, dict) and step.get("tool"):
                    self._record(step["tool"], {}, CapabilityResult(success=bool(step.get("success")),
                                                                    metadata=step.get("metadata") or {}),
                                 None, f"{call_id}:{index}")
            return
        receipt = receipt_for(tool, params, result, expected_edit=expected_edit)
        if receipt is None:
            return
        if tool == "file_delete" and result.success and receipt["tier"] != "rejected":
            path = Path(receipt["target"])
            try:
                rel = str(path.resolve().relative_to(Path.cwd().resolve()))
            except (OSError, ValueError):
                rel = path.name
            self._removed[receipt["target"]] = (rel, str((result.metadata or {}).get("trashed") or ""))
        key = ("file", receipt["target"]) if tool.startswith("file_") else ("call", call_id or str(len(self._receipts)))
        self._put(key, receipt)

    def _record_shell(self, params: dict[str, Any], result: CapabilityResult, call_id: str) -> None:
        meta = result.metadata or {}
        root = Path(params.get("cwd") or Path.cwd()).expanduser().resolve()
        if meta.get("shell_untracked"):
            self._put(("call", call_id or str(len(self._receipts))), {
                "tool": "bash_execute", "target": f"shell in {root}"[:240], "tier": "unknown",
                "evidence": "The command could change files, but no snapshot was available to see what changed."})
            return
        changes = meta.get("shell_changes")
        if not isinstance(changes, dict):
            return
        if removed := [rel for rel in (changes.get("removed") or []) if isinstance(rel, str)]:
            from rune.safety import workspace_snapshot

            # The newest snapshot is the one taken just before this command ran.
            kept = workspace_snapshot.latest(root)
            for rel in removed[:200]:
                self._removed[str(root / rel)] = (rel, str(kept / rel) if kept else "")
        labels = {"created": "Created", "modified": "Modified",
                  "removed": "Removed; a snapshot was taken before the command ran"}
        for kind, label in labels.items():
            paths = [rel for rel in (changes.get(kind) or []) if not _CACHE_DIRS.intersection(Path(rel).parts)]
            for rel in paths[:_MAX_PER_KIND]:
                target = str(root / rel)[:240]
                self._put(("file", target), {"tool": "bash_execute", "target": target, "tier": "observed",
                                             "evidence": f"{label} by a shell command; not compared with requested content."})
            if len(paths) > _MAX_PER_KIND:
                self._put(("call", f"{call_id}:{kind}"), {
                    "tool": "bash_execute", "target": f"{len(paths) - _MAX_PER_KIND} more {kind} under {root}"[:240],
                    "tier": "observed", "evidence": f"{label} by a shell command; listed in the command's result."})

    def removal_review(self, request: str, locale: str = "en") -> tuple[dict[str, Any], str] | None:
        """A check and an answer note for removed files the request never named.

        Approval prompts are accepted almost every time, so removing unnamed user files is not a
        confirmed success. Naming is judged on path text the user wrote, which is language-independent;
        regenerable build and cache output is exempt.
        """
        from rune.safety.recoverable import is_regenerable

        said = request.casefold()

        def named(path: Path) -> bool:
            words = [path.name, *(part for part in path.parent.parts if len(part) > 1)]
            if len(path.suffix) > 1:
                words.append(path.suffix)
            return any(word.casefold() in said for word in words if word and word != "/")

        unnamed = {path: kept for path, (rel, kept) in self._removed.items()
                   if not is_regenerable(Path(rel)) and not named(Path(rel))}
        if not unnamed:
            return None
        listed = sorted(unnamed)
        detail = ("Removed files the request did not name; confirm they were meant to go:\n"
                  + "\n".join(f"- {path}" + (f" (copy: {kept})" if kept else " (no copy kept)")
                               for path, kept in ((p, unnamed[p]) for p in listed[:20]))
                  + (f"\n- and {len(listed) - 20} more" if len(listed) > 20 else ""))
        check = {"required": True, "status": "unverified", "method": "removal_review", "detail": detail,
                 "requirements": [{"kind": "removal", "target": path, "status": "unverified"} for path in listed[:50]]}
        korean = locale.split("-", 1)[0] == "ko"
        lines = "\n".join(f"- {path}" + (f" → {kept}" if kept else "") for path, kept in ((p, unnamed[p]) for p in listed[:20]))
        note = (f"\n\n요청에 이름이 없던 파일 {len(listed)}개를 삭제했습니다. 지워도 되는 파일이었는지 확인해 주세요. "
                f"삭제 전 사본 위치:\n{lines}" if korean else
                f"\n\nRemoved {len(listed)} file(s) the request did not name. Please confirm they were meant to go. "
                f"Copies from before the removal:\n{lines}")
        return check, note

    def finalize(self, requirement_acceptance: dict | None) -> list[dict[str, Any]]:
        """Return receipts, upgrading only the browser save whose stored record was read back."""
        acceptance = requirement_acceptance or {}
        readback = acceptance.get("independentReadback") or {}
        saved = acceptance.get("savedWriteSeqs") if acceptance.get("status") == "pass" and readback.get("status") == "pass" else None
        receipts = []
        for receipt in self._receipts.values():
            if (saved and receipt["tool"] == "browser_act" and receipt["tier"] == "acknowledged"
                    and receipt.get("seqs") == saved):
                receipt = {**receipt, "tier": "verified",
                           "evidence": receipt["evidence"] + "; the stored record was read back and matched."}
            receipts.append({key: value for key, value in receipt.items() if key != "seqs"})
        if self._omitted:
            receipts.append({"tool": "ledger", "target": f"{self._omitted} earlier changes not listed", "tier": "unknown",
                             "evidence": "Older entries were dropped to keep this list short; the run log has every call."})
        return receipts
