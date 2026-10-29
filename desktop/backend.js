'use strict';

const http = require('node:http');
const { performance } = require('node:perf_hooks');

async function waitForBackend(origin, { timeoutMs = 30000, intervalMs = 400, get = http.get } = {}) {
  const deadline = performance.now() + timeoutMs;
  let reason = 'The local engine is not responding.';
  while (performance.now() < deadline) {
    try {
      return await new Promise((resolve, reject) => {
        const req = get(new URL('/ready', origin), res => {
          let body = '';
          res.setEncoding('utf8');
          res.on('data', chunk => {
            body += chunk;
            if (body.length > 65536) req.destroy(new Error('The engine returned an invalid status.'));
          });
          res.on('error', reject);
          res.on('end', () => {
            try {
              const state = JSON.parse(body);
              if (res.statusCode !== 200 || state.ready !== true || !Number.isInteger(state.pid) || state.pid <= 0) {
                throw new Error('The local engine is running, but is not ready yet.');
              }
              resolve(state);
            } catch (error) { reject(error); }
          });
        });
        req.on('error', reject);
        req.setTimeout(Math.max(1, Math.min(2000, deadline - performance.now())),
          () => req.destroy(new Error('The local engine did not respond in time.')));
      });
    } catch (error) { reason = error.message; }
    const remaining = deadline - performance.now();
    if (remaining > 0) await new Promise(resolve => setTimeout(resolve, Math.min(intervalMs, remaining)));
  }
  throw new Error(reason);
}

module.exports = { waitForBackend };
