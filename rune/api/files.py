"""Download files from a conversation's recorded workspace."""

from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from starlette.responses import FileResponse, Response

from rune.api import conversation_wiring


async def download_file(session_id: str, path: str) -> Response:
    from rune.computer.artifacts import Artifacts
    from rune.computer.protocol import DesktopError
    try:
        published = Artifacts().for_path(session_id, path)
    except (OSError, ValueError, KeyError, DesktopError) as exc:
        raise HTTPException(409, "The published download could not be validated.") from exc
    if published is not None:
        receipt, content = published
        return Response(content, media_type="application/octet-stream", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(receipt['name'], safe='')}",
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        })
    workspace = await conversation_wiring.get_workspace(session_id)
    if not workspace:
        raise HTTPException(404, "Conversation workspace not found")
    try:
        root = Path(workspace).resolve()
        target = (root / path).resolve(strict=True)
        if not target.is_relative_to(root):
            raise HTTPException(403, "Path escapes the conversation workspace")
        if not target.is_file():
            raise HTTPException(404, "File not found")
    except (ValueError, RuntimeError):
        raise HTTPException(400, "Invalid file path") from None
    except OSError:
        raise HTTPException(404, "File not found") from None
    return FileResponse(
        target, filename=target.name,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
