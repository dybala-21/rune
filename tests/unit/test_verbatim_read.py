import pytest

from rune.agent.tool_adapter import _format_tool_output
from rune.capabilities.file import FileReadParams, file_read


@pytest.mark.asyncio
async def test_verbatim_read_preserves_actual_markers_and_line_endings(tmp_path):
    path = tmp_path / "original.txt"
    text = "Path: this is real content\r\n     1\talso real\r\n[END: keep this]"
    path.write_bytes(text.encode())
    result = await file_read(FileReadParams(path=str(path), raw=True))
    assert result.success and result.output == text
    assert _format_tool_output("file_read", {"path": str(path)}, result) == text
    partial = await file_read(FileReadParams(path=str(path), raw=True, limit=1))
    assert not partial.success


@pytest.mark.asyncio
async def test_verbatim_read_bypasses_annotated_cache_and_checks_arguments(tmp_path):
    from rune.agent.cognitive_cache import SessionToolCache
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.file import register_file_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    source = tmp_path / "original.txt"
    text = "Path: 원문\r\n     1\tkeep this\r\n[END: literal]"
    source.write_bytes(text.encode())
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    cache = SessionToolCache()
    reader = build_tool_set(ToolAdapterOptions(
        allowed_tools=["file_read"], cognitive_cache=cache, workspace_root=str(tmp_path),
    ), registry)["file_read"].function
    annotated = await reader(path="original.txt")
    assert annotated != text and cache.entry_count == 1
    for _ in range(2):
        assert await reader(path="original.txt", raw=True) == text
    assert cache.hit_count == 0
    assert "[ERROR]" in await reader(path="original.txt", raw=True, limit=1)
    assert "[ERROR]" in await reader(path="original.txt", maxSize=1)
    source.write_bytes(b"caf\xe9")
    assert "caf\xe9" in await reader(path="original.txt", encoding="latin-1")
    source.write_text("changed outside Rune")
    assert await reader(path="original.txt", raw=True) == "changed outside Rune"
