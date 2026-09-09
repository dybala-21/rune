"""Read and publish one version of an office bundle."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock


class BundleError(ValueError):
    def __init__(self, code: str, message: str, current_revision: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.current_revision = current_revision


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _bytes(path: Path) -> bytes:
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise BundleError("invalid_manifest", "Invalid bundle metadata file")
    with path.open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise BundleError("invalid_manifest", "Bundle metadata exceeds 16 MiB")
    return data


def _read(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise BundleError("invalid_manifest", "Bundle metadata must be an object")
    return value


def check_controls(root: Path) -> None:
    for name in ("versions", "current.json", ".publish.lock"):
        if (root / name).is_symlink():
            raise BundleError("invalid_path", "Bundle control paths must not be symlinks")


@dataclass(frozen=True)
class BundleSnapshot:
    revision: str
    manifest: dict[str, Any]
    manifest_sha256: str
    pointer_sha256: str | None
    changed_artifacts: tuple[str, ...]


def read_snapshot(root: Path, revision: str | None = None, *, strict: bool = True) -> BundleSnapshot | None:
    check_controls(root)
    pointer_path = root / "current.json"
    pointer_bytes = _bytes(pointer_path) if pointer_path.exists() else None
    pointer = _read(pointer_bytes) if pointer_bytes is not None else None
    selected = revision or (pointer or {}).get("revision")
    if selected is None:
        return None
    if not isinstance(selected, str) or len(selected) != 32 or any(c not in "0123456789abcdef" for c in selected):
        raise BundleError("invalid_revision", "Invalid bundle revision")
    version = root / "versions" / selected
    if version.is_symlink():
        raise BundleError("invalid_path", "Bundle versions must not be symlinks")
    path = version / "manifest.json"
    current = pointer is not None and pointer.get("revision") == selected
    if current and pointer is not None and Path(pointer.get("manifest", "")).resolve() != path:
        raise BundleError("invalid_manifest", "Pointer manifest path is outside its revision")
    manifest_bytes = _bytes(path)
    manifest = _read(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("schema_version") not in (1, 2) or manifest.get("revision") != selected:
        raise BundleError("invalid_manifest", "Unsupported or mismatched bundle manifest")
    if current and pointer is not None and manifest.get("schema_version") == 2 and pointer.get("manifest_sha256") != manifest_hash:
        raise BundleError("manifest_modified", "Current manifest hash differs from its pointer")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 12:
        raise BundleError("invalid_manifest", "Bundle artifact list is invalid")
    changed = []
    names: set[str] = set()
    for artifact in artifacts:
        artifact_path = Path(artifact["path"])
        name = artifact_path.name
        if (artifact_path != version / name or name.casefold() in names or name.startswith(".")
                or artifact_path.is_symlink()):
            raise BundleError("invalid_path", "Artifact path is outside its revision or duplicated")
        names.add(name.casefold())
        if not artifact_path.is_file() or digest(artifact_path) != artifact.get("sha256"):
            changed.append(name)
    if strict and changed:
        raise BundleError("artifact_modified", "Saved artifacts were modified: " + ", ".join(changed), selected)
    return BundleSnapshot(selected, manifest, manifest_hash,
                          hashlib.sha256(pointer_bytes).hexdigest() if pointer_bytes else None,
                          tuple(changed))


def publish(root: Path, stage: Path, manifest: dict[str, Any], base: BundleSnapshot | None) -> None:
    check_controls(root)
    with FileLock(str(root / ".publish.lock"), timeout=1):
        current = read_snapshot(root)
        expected = base.pointer_sha256 if base else None
        actual = current.pointer_sha256 if current else None
        if actual != expected:
            raise BundleError("revision_conflict", "Bundle changed during rendering; inspect before retrying",
                              current.revision if current else None)
        version = stage / "version"
        manifest_path = version / "manifest.json"
        manifest_path.write_text(encode(manifest), encoding="utf-8")
        final = root / "versions" / manifest["revision"]
        pointer = stage / "current.json"
        pointer.write_text(encode({"revision": manifest["revision"],
                                   "manifest": str(final / "manifest.json"),
                                   "manifest_sha256": digest(manifest_path)}), encoding="utf-8")
        version.rename(final)
        os.replace(pointer, root / "current.json")


def changes_between(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    spec_before, spec_after = before.get("spec", {}), after["spec"]
    metrics_before, metrics_after = before.get("metrics", {}), after["metrics"]
    return {
        "spec": {key: {"before": spec_before.get(key), "after": spec_after.get(key)}
                 for key in spec_before.keys() | spec_after.keys() if spec_before.get(key) != spec_after.get(key)},
        "metrics": {key: {"before": metrics_before.get(key), "after": metrics_after.get(key)}
                    for key in metrics_before.keys() | metrics_after.keys()
                    if metrics_before.get(key) != metrics_after.get(key)},
        "files": [a["path"] for a in after["artifacts"]],
    }
