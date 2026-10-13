"""Check approval persistence and duplicate submissions across API restarts."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

_WORKER = '''
import asyncio, sys
from pathlib import Path
import pytest, uvicorn
from fastapi import FastAPI
from tests.unit import test_routine_approval as fixture
from rune.types import CapabilityResult
root = Path(sys.argv[1])
patch = pytest.MonkeyPatch()
context = fixture.task.__wrapped__(root, patch)
task = next(context)
async def submit(params):
    with (root / 'submissions.txt').open('a') as handle:
        handle.write(params['body'] + '\\n')
    return CapabilityResult(success=True, output='accepted')
task.send.side_effect = submit
if not task.store.list_cron_jobs():
    identity = fixture.create(task)
    asyncio.run(fixture.tick(task, identity))
app = FastAPI()
app.include_router(fixture.api.router)
uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[2]), log_level='error')
'''


def test_approval_survives_api_restart_and_duplicate_submission(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    process = None
    log = (tmp_path / 'worker.log').open('w')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, 'RUNE_HOME': str(tmp_path / 'state'), 'RUNE_WORKSPACE': str(tmp_path),
                   'RUNE_ISOLATION_ROOT': str(tmp_path), 'RUNE_REQUIRE_TOKEN': '0'}
    with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
        def start():
            nonlocal process
            process = subprocess.Popen([sys.executable, '-c', _WORKER, str(tmp_path), str(port)],
                                       cwd=repo, env=environment, stdout=log, stderr=log)
            for _ in range(150):
                assert process.poll() is None, (tmp_path / 'worker.log').read_text()
                try:
                    response = client.get('/cron')
                    if response.status_code == 200:
                        return response.json()['jobs'][0]
                except httpx.TransportError:
                    time.sleep(.05)
            raise AssertionError('Approval API did not start')

        def stop():
            process.terminate()
            process.wait(timeout=10)

        try:
            first = start()
            operation = first['recentRuns'][0]['id']
            assert first['recentRuns'][0]['state'] == 'needs_approval'
            stop()
            second = start()
            assert second['recentRuns'][0]['id'] == operation
            assert second['budget'] == first['budget']
            route = f"/cron/{first['id']}/approval"
            response = client.post(route, json={'operationId': operation, 'approved': True})
            assert response.status_code == 200 and response.json()['status'] == 'verified', response.text
            stop()
            third = start()
            repeated = client.post(route, json={'operationId': operation, 'approved': True})
            assert repeated.json() == response.json()
            assert client.get('/cron').json()['jobs'][0]['budget'] == third['budget']
            assert (tmp_path / 'submissions.txt').read_text() == 'draft=Source facts\n'
        finally:
            if process is not None and process.poll() is None:
                stop()
            log.close()
