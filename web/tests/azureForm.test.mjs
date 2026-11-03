import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createServer } from 'vite';

let server, connectionErrors, rowErrors;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ connectionErrors, rowErrors } = await server.ssrLoadModule('/src/components/azureForm.ts'));
});
after(async () => { await server?.close(); });
const connection = { endpoint: 'https://fixture.services.ai.azure.com/openai/v1', hasKey: true, keyScope: 'project' };

test('a new resource requires its key before checking or saving', () => {
  assert.deepEqual(connectionErrors(connection, connection.endpoint, ''), {});
  assert.deepEqual(connectionErrors(connection, 'https://fixture.services.ai.azure.com/', ''), {});
  assert.ok(connectionErrors(connection, 'https://other.services.ai.azure.com', '')['azure-key']);
  assert.deepEqual(connectionErrors(connection, 'https://other.services.ai.azure.com', 'new-key'), {});
  assert.ok(connectionErrors(connection, connection.endpoint, 'bad\nkey')['azure-key']);
});

test('the form refuses unsafe URLs that the diagnostic also refuses', () => {
  for (const endpoint of ['http://fixture.services.ai.azure.com', 'https://fixture.services.ai.azure.com.evil.test',
    'https://user:secret@fixture.services.ai.azure.com', connection.endpoint + '?key=x', connection.endpoint + '#fragment',
    'https://fixture.services.ai.azure.com:8443', 'https://127.0.0.1']) {
    assert.ok(connectionErrors(connection, endpoint, 'test-key')['azure-endpoint']);
  }
});

test('row validation identifies both duplicates and the exact invalid field', () => {
  const rows = [{ id: 'a', name: 'daily', model: 'Kimi-K2.7-Code' }, { id: 'b', name: ' daily ', model: '' }];
  const errors = rowErrors(rows);
  assert.deepEqual(Object.keys(errors).sort(), ['a-name', 'b-model', 'b-name']);
  assert.deepEqual(rowErrors([{ id: 'a', name: 'daily', model: 'Kimi-K2.7-Code' }]), {});
  assert.ok(rowErrors([{ id: 'a', name: 'invalid/path', model: 'DeepSeek-V4-Pro' }])['a-name']);
});
