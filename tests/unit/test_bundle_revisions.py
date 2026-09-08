"""Bundle updates preserve unspecified fields and reject stale publications."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rune.capabilities.bundle_revision import digest
from rune.capabilities.document_bundle import (
    BundleInspectParams,
    BundleUpdateParams,
    document_bundle,
    document_bundle_inspect,
    document_bundle_update,
)
from tests.unit.test_document_bundle import office as office


def update_request(office, revision, **changes):
    return BundleUpdateParams(directory=office.directory, base_revision=revision,
                              expected_source_sha256=digest(Path(office.source_path)), changes=changes)


async def test_partial_update_preserves_documents_and_records_changes(office):
    initial = await document_bundle(office)
    inspected = await document_bundle_inspect(BundleInspectParams(directory=office.directory))
    assert inspected.success, inspected.error
    info = json.loads(inspected.output)
    assert info['current_source_sha256'] == digest(Path(office.source_path))
    updated = await document_bundle_update(update_request(
        office, info['revision'], filters=[{'column': '상태', 'operator': 'eq', 'values': ['확정']}],
        documents=[{'filename': 'briefing.docx', 'title': '확정 주문 요약'}],
    ))
    assert updated.success, updated.error
    result = json.loads(updated.output)
    assert result['metrics'] == {'amount': 23_000_000, 'orders': 3}
    assert result['parent_revision'] == initial.metadata['revision']
    assert result['changes']['metrics']['amount'] == {'before': 34_000_000, 'after': 23_000_000}
    manifest = json.loads(Path(updated.metadata['manifest']).read_text())
    assert manifest['spec']['documents'][0] == info['spec']['documents'][0]
    assert manifest['spec']['documents'][1]['title'] == '확정 주문 요약'
    assert updated.metadata['receipt']['checks']['task_acceptance'] == 'not_performed'
    assert len(list((Path(office.directory) / 'versions').iterdir())) == 2
    stale = await document_bundle_update(update_request(office, info['revision'], filters=[]))
    assert not stale.success and stale.metadata['code'] == 'revision_conflict'


async def test_two_simultaneous_updates_publish_once(office, monkeypatch):
    import rune.capabilities.document_bundle as bundle

    initial = await document_bundle(office)
    revision = initial.metadata['revision']
    render = bundle._render_staged
    arrived = 0
    ready = asyncio.Event()

    async def synchronized(payload):
        nonlocal arrived
        result = await render(payload)
        arrived += 1
        if arrived == 2:
            ready.set()
        await ready.wait()
        return result

    monkeypatch.setattr(bundle, '_render_staged', synchronized)
    a, b = await asyncio.gather(
        document_bundle_update(update_request(office, revision, filters=[])),
        document_bundle_update(update_request(office, revision, documents=[{'filename': 'briefing.docx', 'title': 'New title'}])),
    )
    assert a.success != b.success
    assert (b if a.success else a).metadata['code'] == 'revision_conflict'
    assert len(list((Path(office.directory) / 'versions').iterdir())) == 2


async def test_changed_source_and_artifact_are_not_overwritten(office):
    initial = await document_bundle(office)
    request = update_request(office, initial.metadata['revision'], filters=[])
    source = Path(office.source_path)
    source.write_bytes(source.read_bytes() + b'\n')
    result = await document_bundle_update(request)
    assert result.metadata['code'] == 'source_changed'
    inspected = json.loads((await document_bundle_inspect(BundleInspectParams(directory=office.directory))).output)
    assert inspected['source_stale']
    manifest = json.loads(Path(initial.metadata['manifest']).read_text())
    artifact = Path(manifest['artifacts'][0]['path'])
    artifact.write_bytes(b'user edited file')
    result = await document_bundle_update(update_request(office, initial.metadata['revision'], filters=[]))
    assert result.metadata['code'] == 'artifact_modified'
    assert artifact.read_bytes() == b'user edited file'


async def test_cancel_during_render_preserves_current_and_removes_stage(office, monkeypatch):
    import rune.capabilities.document_bundle as bundle

    initial = await document_bundle(office)
    pointer = Path(office.directory) / 'current.json'
    before = pointer.read_bytes()
    started = asyncio.Event()
    render = bundle._render_staged

    async def notify_start(payload):
        started.set()
        return await render(payload)

    monkeypatch.setattr(bundle, '_render_staged', notify_start)
    task = asyncio.create_task(document_bundle_update(update_request(office, initial.metadata['revision'], filters=[])))
    await started.wait()
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pointer.read_bytes() == before
    assert not list(pointer.parent.glob('.stage-*'))


async def test_manifest_symlink_cannot_redirect_inspection(office, tmp_path):
    initial = await document_bundle(office)
    manifest = Path(initial.metadata['manifest'])
    outside = tmp_path / 'outside.json'
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(outside)
    inspected = await document_bundle_inspect(BundleInspectParams(directory=office.directory))
    assert not inspected.success


async def test_legacy_manifest_upgrades_without_changing_old_version(office):
    first = await document_bundle(office)
    path = Path(first.metadata['manifest'])
    manifest = json.loads(path.read_text())
    manifest['schema_version'] = 1
    path.write_text(json.dumps(manifest))
    pointer = Path(office.directory) / 'current.json'
    data = json.loads(pointer.read_text())
    data.pop('manifest_sha256')
    pointer.write_text(json.dumps(data))
    before = path.read_bytes()
    result = await document_bundle_update(update_request(office, first.metadata['revision'], filters=[]))
    assert result.success, result.error
    assert json.loads(Path(result.metadata['manifest']).read_text())['schema_version'] == 2
    assert path.read_bytes() == before


async def test_inspect_and_update_follow_adapter_paths_and_permissions(office, tmp_path):
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.document_bundle import register_document_bundle_capability
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import get_allowed_tools

    initial = await document_bundle(office)
    registry = CapabilityRegistry()
    register_document_bundle_capability(registry)
    assert 'document_bundle_inspect' in get_allowed_tools('safe')
    assert 'document_bundle_update' not in get_allowed_tools('safe')
    adapted = build_tool_set(ToolAdapterOptions(workspace_root=str(tmp_path), enable_guardian=False), registry=registry)
    inspected = await adapted['document_bundle_inspect'].function(directory='bundle')
    assert initial.metadata['revision'] in inspected
    request = update_request(office, initial.metadata['revision'], source_path='sales.csv', filters=[]).model_dump(exclude_unset=True)
    request['directory'] = 'bundle'
    response = await adapted['document_bundle_update'].function(**request)
    assert 'revision' in response
    current = json.loads((Path(office.directory) / 'current.json').read_text())
    assert current['revision'] != initial.metadata['revision']


def test_artifact_receipt_does_not_certify_whole_task():
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    receipt = {'kind': 'document_bundle', 'revision': 'a' * 32,
               'checks': {'native_content': 'pass', 'task_acceptance': 'not_performed'}}
    trace = CompletionTrace(reason='completed', artifact_receipts=[receipt])
    payload = build_trust_payload(trace)
    assert payload['artifactReceipts'] == [receipt]
    assert payload['verified'] is False


async def test_disabling_updates_preserves_published_versions(office, monkeypatch):
    from rune.capabilities.document_bundle import register_document_bundle_capability
    from rune.capabilities.registry import CapabilityRegistry

    initial = await document_bundle(office)
    pointer = Path(office.directory) / 'current.json'
    before = pointer.read_bytes()
    monkeypatch.delenv('RUNE_BUNDLE_UPDATE_ENABLED')
    registry = CapabilityRegistry()
    register_document_bundle_capability(registry)
    assert registry.get('document_bundle_update') is None
    assert registry.get('document_bundle_inspect') is not None
    result = await document_bundle_update(update_request(office, initial.metadata['revision'], filters=[]))
    assert result.metadata['code'] == 'feature_disabled'
    assert pointer.read_bytes() == before
    assert (await document_bundle_inspect(BundleInspectParams(directory=office.directory))).success


@pytest.mark.parametrize('failure', [RuntimeError('later step failed'), asyncio.CancelledError()])
async def test_run_retains_completed_artifact_checks_after_interruption(monkeypatch, tmp_path, failure):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    monkeypatch.setenv('RUNE_HOME', str(tmp_path))
    monkeypatch.setenv('RUNE_AUTO_SKILL', '0')
    loop = NativeAgentLoop()
    receipt = {'kind': 'document_bundle', 'revision': 'a' * 32}

    async def prompt(*args):
        return 'test'

    async def execute(**kwargs):
        loop._artifact_receipts.append(receipt)
        raise failure

    monkeypatch.setattr(loop, '_build_system_prompt', prompt)
    monkeypatch.setattr(loop, '_select_tools', lambda _: [])
    monkeypatch.setattr(loop, '_execute_loop', execute)
    classification = ClassificationResult(goal_type='full', confidence=1, tier=1)
    trace = await loop.run('Create and revise documents', classification=classification)
    assert trace.artifact_receipts == [receipt]
    assert build_trust_payload(trace)['verified'] is False

    async def completed(**kwargs):
        return CompletionTrace(reason='completed')

    monkeypatch.setattr(loop, '_execute_loop', completed)
    next_trace = await loop.run('A different request', classification=classification)
    assert next_trace.artifact_receipts == []
