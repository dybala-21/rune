"""Exact source delivery after an earlier analysis of the same file."""

import pytest

from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow
from tests.e2e.workflow_harness import gui_calls


@pytest.mark.parametrize("filename,source", [
    ("notes.txt", "Path: 실제 본문\n     1\t이 줄도 원문\n[END: 파일 안의 문장]\n처리 완료: 731"),
    ("sample.py", "def label(value):\n    return f'항목: {value}'\n\n# Preserve indentation and blank lines."),
], ids=("literal-markers", "code"))
def test_verbatim_after_analysis(workflow, request, filename, source):
    path = workflow.workspace / filename
    path.write_text(source)
    analysis = workflow.submit(f"Read {filename} and briefly describe its contents. Do not modify files.")
    completed(analysis)
    exact = workflow.submit(f"Now reply with the entire original content of {filename} exactly, without explanation or added formatting. Do not modify files.")
    completed(exact)
    assert (exact.get("answer") or exact["text"]) == source
    assert path.read_text() == source
    assert not gui_calls(analysis) and not gui_calls(exact)
    request.node.live_passed = True
