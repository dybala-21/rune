#!/usr/bin/env sh
set -eu

OUTPUT_DIR="${1:-/tmp/rune-wheelhouse}"
mkdir -p "$OUTPUT_DIR"

python3 -m pip wheel --wheel-dir "$OUTPUT_DIR" .

python3 - "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


files = [
    {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    for path in sorted(output_dir.glob("*.whl"))
]

directory_digest = hashlib.sha256()
for row in files:
    directory_digest.update(row["path"].encode("utf-8"))
    directory_digest.update(b"\0")
    directory_digest.update(row["sha256"].encode("ascii"))
    directory_digest.update(b"\0")

manifest = {
    "created_at": datetime.now(UTC).isoformat(),
    "git_sha": os.environ.get("RUNE_BENCH_SOURCE_GIT_SHA") or run(["git", "rev-parse", "HEAD"]),
    "git_status_porcelain": run(["git", "status", "--porcelain", "--untracked-files=all"]),
    "platform": platform.platform(),
    "source_diff_sha256": os.environ.get("RUNE_BENCH_SOURCE_DIFF_SHA256"),
    "wheelhouse_sha256": directory_digest.hexdigest(),
    "files": files,
}

(output_dir / "wheelhouse-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(manifest["wheelhouse_sha256"])
PY
