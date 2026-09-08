"""Exercise the embedding provider through a real child process."""

from __future__ import annotations

import asyncio
import os
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rune.llm.embedding_models import EMBEDDING_MODELS
from rune.llm.local_embedding import LocalEmbeddingEngine
from rune.utils.process_worker import (
    ProcessWorker,
    WorkerUnavailable,
    receive_message,
    send_message,
)

pytestmark = pytest.mark.embedding_engine


def fake_worker(sock: socket.socket) -> None:
    count = 0
    try:
        while True:
            req = receive_message(sock, None)
            params = req['payload']
            action = params.get('action')
            if action == 'crash':
                os._exit(19)
            if action == 'partial':
                sock.sendall(struct.pack('!I', 100) + b'{')
                time.sleep(30)
            if action == 'hang':
                time.sleep(30)
            time.sleep(params.get('delay', 0))
            count += 1
            result = {'count': count, 'dimensions': 768, 'fingerprint': 'a' * 64,
                      'vectors': [[float(count)] * 768 for _ in params.get('texts', [])]}
            send_message(sock, {'id': 'bad' if action == 'wrong_id' else req['id'],
                                'result': result}, None)
    except (EOFError, OSError):
        return
    finally:
        sock.close()


@pytest.fixture
def worker():
    w = ProcessWorker(fake_worker, cooldown=0)
    yield w
    w.close()


@pytest.fixture
def engine(tmp_path, monkeypatch, worker):
    (tmp_path / EMBEDDING_MODELS['nomic-embed-text'].file).write_bytes(b'fixture')
    monkeypatch.setattr('rune.llm.local_embedding._models_dir', lambda: tmp_path)
    eng = LocalEmbeddingEngine(worker=worker)
    yield eng
    eng.dispose()


def test_sync_serialization(engine):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(engine.embed_sync, ['text'] * 16))
    assert sorted(v[0] for v in values) == list(range(1, 17))
    assert engine.is_ready()
    assert engine.fingerprint == 'a' * 64


async def test_async_and_sync_calls_share_worker(engine):
    results = await asyncio.gather(
        engine.embed(['a', 'b']), engine.embed_single('c'),
        asyncio.to_thread(engine.embed_sync, 'd'),
    )
    assert sorted([results[0][0][0], results[1][0], results[2][0]]) == [1, 2, 3]
    assert results[0][0] == results[0][1]


@pytest.mark.parametrize('action', ['crash', 'partial', 'hang', 'wrong_id'])
def test_failure_reaps_worker_and_next_request_recovers(worker, action):
    worker.request({})
    pid = worker.status['pid']
    start = time.monotonic()
    with pytest.raises(WorkerUnavailable):
        worker.request({'action': action}, timeout=0.2)
    assert time.monotonic() - start < 2
    assert worker.status['pid'] is None
    assert worker.status['state'] == 'degraded'
    assert worker.request({})['count'] == 1
    assert worker.status['pid'] != pid


def test_cooldown_prevents_restart_loop():
    with_worker = ProcessWorker(fake_worker, cooldown=60)
    try:
        with pytest.raises(WorkerUnavailable):
            with_worker.request({'action': 'crash'})
        generation = with_worker.generation
        with pytest.raises(WorkerUnavailable):
            with_worker.request({})
        assert with_worker.generation == generation
    finally:
        with_worker.close()


async def test_close_interrupts_hung_request(worker):
    worker.request({})
    task = asyncio.create_task(asyncio.to_thread(worker.request, {'action': 'hang'}))
    await asyncio.sleep(0.05)
    await asyncio.wait_for(asyncio.to_thread(worker.close), timeout=2)
    with pytest.raises(WorkerUnavailable):
        await task
    assert worker.status['pid'] is None


async def test_cancelled_queued_embedding_is_not_sent(engine, worker):
    blocker = asyncio.create_task(asyncio.to_thread(worker.request, {'delay': 0.2}))
    await asyncio.sleep(0.05)
    first = asyncio.create_task(engine.embed_single('first'))
    await asyncio.sleep(0.01)
    cancelled = asyncio.create_task(engine.embed_single('cancelled'))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await blocker
    await first
    assert (await engine.embed_single('last'))[0] == 3


def test_dispose_rejects_new_work(engine):
    engine.embed_sync('init')
    engine.dispose()
    assert engine.health['pid'] is None
    with pytest.raises(WorkerUnavailable):
        engine.embed_sync('after close')


def test_missing_model_does_not_block_search(tmp_path, monkeypatch):
    monkeypatch.setattr('rune.llm.local_embedding._models_dir', lambda: tmp_path)
    monkeypatch.setattr(LocalEmbeddingEngine, '_background_download', lambda self, model: (True, time.monotonic()))
    engine = LocalEmbeddingEngine()
    try:
        with pytest.raises(WorkerUnavailable, match='not cached'):
            engine.embed_sync('query')
        assert engine.health['pid'] is None
    finally:
        engine.dispose()
