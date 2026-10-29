'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { isIP } = require('node:net');

function readEndpoint(env = process.env) {
  const home = (env.RUNE_HOME || path.join(os.homedir(), '.rune'))
    .replace(/^~(?=\/|$)/, os.homedir()).replace(/\$\{(\w+)\}|\$(\w+)/g, (match, a, b) => env[a || b] ?? match);
  let saved = { host: '127.0.0.1', port: 18789 };
  try { saved = JSON.parse(fs.readFileSync(path.join(home, 'daemon.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw new Error('Could not read the Rune daemon address.'); }
  const host = env.RUNE_API_HOST ?? saved.host;
  const port = Number(env.RUNE_API_PORT ?? saved.port);
  if (typeof host !== 'string' || (host !== 'localhost' && !isIP(host))
      || typeof saved.port === 'boolean' || !Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('The Rune daemon address or port is invalid.');
  }
  const target = ({ '0.0.0.0': '127.0.0.1', '::': '::1' })[host] || host;
  return { host, port, origin: `http://${isIP(target) === 6 ? `[${target}]` : target}:${port}` };
}

module.exports = { readEndpoint };
