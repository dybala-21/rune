import { useCallback, useEffect, useState } from 'react';
import {
  fetchMCPServers,
  addMCPServer,
  deleteMCPServer,
  testMCPServer,
  type MCPServerInfo,
  type MCPTestResult,
} from '../api';

interface MCPPanelProps {
  onClose: () => void;
}

type ViewMode = 'list' | 'add';
type TransportType = 'stdio' | 'sse' | 'streamable-http';

const TRANSPORT_LABELS: Record<TransportType, string> = {
  stdio: 'Local Command (stdio)',
  sse: 'Server-Sent Events (SSE)',
  'streamable-http': 'Streamable HTTP',
};

export function MCPPanel({ onClose }: MCPPanelProps) {
  const [servers, setServers] = useState<MCPServerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [testResults, setTestResults] = useState<Record<string, MCPTestResult>>({});
  const [testing, setTesting] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Add form state
  const [formName, setFormName] = useState('');
  const [formTransport, setFormTransport] = useState<TransportType>('stdio');
  const [formCommand, setFormCommand] = useState('');
  const [formArgs, setFormArgs] = useState('');
  const [formUrl, setFormUrl] = useState('');

  const loadServers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMCPServers();
      setServers(data.servers);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadServers(); }, [loadServers]);

  const handleTest = async (name: string) => {
    setTesting(name);
    try {
      const result = await testMCPServer(name);
      setTestResults(prev => ({ ...prev, [name]: result }));
    } catch {
      setTestResults(prev => ({
        ...prev,
        [name]: { name, success: false, message: 'Test request failed', tools_count: 0 },
      }));
    } finally {
      setTesting(null);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete MCP server "${name}"?`)) return;
    try {
      await deleteMCPServer(name);
      await loadServers();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    }
  };

  const handleAdd = async () => {
    if (!formName.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await addMCPServer({
        name: formName.trim(),
        transport: formTransport,
        command: formTransport === 'stdio' ? formCommand.trim() || undefined : undefined,
        args: formTransport === 'stdio' && formArgs.trim()
          ? formArgs.split(/\s+/)
          : [],
        url: formTransport !== 'stdio' ? formUrl.trim() || undefined : undefined,
      });
      setFormName('');
      setFormCommand('');
      setFormArgs('');
      setFormUrl('');
      setViewMode('list');
      await loadServers();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Add failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{
      position: 'fixed', top: 0, right: 0, bottom: 0,
      width: 480, background: 'var(--bg-secondary, #1a1a2e)',
      borderLeft: '1px solid var(--border, #333)',
      display: 'flex', flexDirection: 'column',
      zIndex: 100, boxShadow: '-4px 0 20px rgba(0,0,0,0.3)',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px', borderBottom: '1px solid var(--border, #333)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16, color: 'var(--text, #e0e0e0)' }}>
            MCP Servers
          </h2>
          <span style={{ fontSize: 12, color: 'var(--text-muted, #888)' }}>
            {servers.length} server{servers.length !== 1 ? 's' : ''} configured
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {viewMode === 'list' && (
            <button
              onClick={() => setViewMode('add')}
              style={{
                background: 'var(--accent, #61afef)', color: '#fff',
                border: 'none', borderRadius: 6, padding: '6px 14px',
                cursor: 'pointer', fontSize: 13, fontWeight: 500,
              }}
            >
              + Add Server
            </button>
          )}
          <button
            onClick={onClose}
            style={{
              background: 'transparent', border: 'none',
              color: 'var(--text-muted, #888)', cursor: 'pointer',
              fontSize: 18, padding: '4px 8px',
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: '8px 20px', background: 'rgba(224, 108, 117, 0.1)',
          color: 'var(--danger, #e06c75)', fontSize: 13,
        }}>
          {error}
        </div>
      )}

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '12px 20px' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted, #888)' }}>
            Loading...
          </div>
        ) : viewMode === 'add' ? (
          /* Add form */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <button
              onClick={() => { setViewMode('list'); setError(null); }}
              style={{
                background: 'transparent', border: 'none',
                color: 'var(--accent, #61afef)', cursor: 'pointer',
                fontSize: 13, textAlign: 'left', padding: 0,
              }}
            >
              ← Back to list
            </button>

            {/* Name */}
            <label style={{ fontSize: 13, color: 'var(--text-muted, #888)' }}>
              Server Name
              <input
                value={formName}
                onChange={e => setFormName(e.target.value)}
                placeholder="my-mcp-server"
                style={{
                  width: '100%', marginTop: 4, padding: '8px 12px',
                  background: 'var(--bg-primary, #0d1117)',
                  border: '1px solid var(--border, #333)',
                  borderRadius: 6, color: 'var(--text, #e0e0e0)',
                  fontSize: 14, outline: 'none',
                }}
              />
            </label>

            {/* Transport */}
            <label style={{ fontSize: 13, color: 'var(--text-muted, #888)' }}>
              Transport
              <select
                value={formTransport}
                onChange={e => setFormTransport(e.target.value as TransportType)}
                style={{
                  width: '100%', marginTop: 4, padding: '8px 12px',
                  background: 'var(--bg-primary, #0d1117)',
                  border: '1px solid var(--border, #333)',
                  borderRadius: 6, color: 'var(--text, #e0e0e0)',
                  fontSize: 14, outline: 'none',
                }}
              >
                {Object.entries(TRANSPORT_LABELS).map(([val, label]) => (
                  <option key={val} value={val}>{label}</option>
                ))}
              </select>
            </label>

            {/* stdio fields */}
            {formTransport === 'stdio' && (
              <>
                <label style={{ fontSize: 13, color: 'var(--text-muted, #888)' }}>
                  Command
                  <input
                    value={formCommand}
                    onChange={e => setFormCommand(e.target.value)}
                    placeholder="npx -y @modelcontextprotocol/server-filesystem"
                    style={{
                      width: '100%', marginTop: 4, padding: '8px 12px',
                      background: 'var(--bg-primary, #0d1117)',
                      border: '1px solid var(--border, #333)',
                      borderRadius: 6, color: 'var(--text, #e0e0e0)',
                      fontSize: 14, outline: 'none',
                    }}
                  />
                </label>
                <label style={{ fontSize: 13, color: 'var(--text-muted, #888)' }}>
                  Arguments (space-separated)
                  <input
                    value={formArgs}
                    onChange={e => setFormArgs(e.target.value)}
                    placeholder="/path/to/allowed/directory"
                    style={{
                      width: '100%', marginTop: 4, padding: '8px 12px',
                      background: 'var(--bg-primary, #0d1117)',
                      border: '1px solid var(--border, #333)',
                      borderRadius: 6, color: 'var(--text, #e0e0e0)',
                      fontSize: 14, outline: 'none',
                    }}
                  />
                </label>
              </>
            )}

            {/* SSE / streamable-http fields */}
            {formTransport !== 'stdio' && (
              <label style={{ fontSize: 13, color: 'var(--text-muted, #888)' }}>
                Server URL
                <input
                  value={formUrl}
                  onChange={e => setFormUrl(e.target.value)}
                  placeholder="http://localhost:3001/sse"
                  style={{
                    width: '100%', marginTop: 4, padding: '8px 12px',
                    background: 'var(--bg-primary, #0d1117)',
                    border: '1px solid var(--border, #333)',
                    borderRadius: 6, color: 'var(--text, #e0e0e0)',
                    fontSize: 14, outline: 'none',
                  }}
                />
              </label>
            )}

            <button
              onClick={handleAdd}
              disabled={saving || !formName.trim()}
              style={{
                background: saving ? 'var(--text-muted, #888)' : 'var(--accent, #61afef)',
                color: '#fff', border: 'none', borderRadius: 6,
                padding: '10px 16px', cursor: saving ? 'default' : 'pointer',
                fontSize: 14, fontWeight: 500, marginTop: 8,
              }}
            >
              {saving ? 'Adding...' : 'Add Server'}
            </button>
          </div>
        ) : servers.length === 0 ? (
          /* Empty state */
          <div style={{
            textAlign: 'center', padding: '60px 20px',
            color: 'var(--text-muted, #888)',
          }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>🔌</div>
            <div style={{ fontSize: 15, marginBottom: 8 }}>No MCP servers configured</div>
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>
              MCP servers extend RUNE with external tools.<br />
              Add a server to get started.
            </div>
          </div>
        ) : (
          /* Server list */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {servers.map(server => {
              const testResult = testResults[server.name];
              return (
                <div
                  key={server.name}
                  style={{
                    padding: '12px 16px',
                    background: 'var(--bg-primary, #0d1117)',
                    borderRadius: 8,
                    border: '1px solid var(--border, #333)',
                  }}
                >
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: 6,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: server.disabled
                          ? 'var(--text-muted, #888)'
                          : 'var(--success, #98c379)',
                        display: 'inline-block',
                      }} />
                      <span style={{
                        fontWeight: 600, fontSize: 14,
                        color: 'var(--text, #e0e0e0)',
                      }}>
                        {server.name}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button
                        onClick={() => handleTest(server.name)}
                        disabled={testing === server.name}
                        style={{
                          background: 'transparent',
                          border: '1px solid var(--border, #333)',
                          borderRadius: 4, padding: '3px 10px',
                          color: 'var(--accent, #61afef)',
                          cursor: 'pointer', fontSize: 12,
                        }}
                      >
                        {testing === server.name ? '...' : 'Test'}
                      </button>
                      <button
                        onClick={() => handleDelete(server.name)}
                        style={{
                          background: 'transparent',
                          border: '1px solid var(--border, #333)',
                          borderRadius: 4, padding: '3px 10px',
                          color: 'var(--danger, #e06c75)',
                          cursor: 'pointer', fontSize: 12,
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  {/* Server details */}
                  <div style={{ fontSize: 12, color: 'var(--text-muted, #888)', lineHeight: 1.6 }}>
                    <span style={{
                      background: 'var(--bg-secondary, #1a1a2e)',
                      padding: '2px 6px', borderRadius: 3,
                      fontSize: 11, marginRight: 8,
                    }}>
                      {server.transport}
                    </span>
                    {server.command && <code>{server.command}</code>}
                    {server.url && <code>{server.url}</code>}
                  </div>

                  {/* Test result */}
                  {testResult && (
                    <div style={{
                      marginTop: 8, padding: '6px 10px', borderRadius: 4,
                      fontSize: 12,
                      background: testResult.success
                        ? 'rgba(152, 195, 121, 0.1)'
                        : 'rgba(224, 108, 117, 0.1)',
                      color: testResult.success
                        ? 'var(--success, #98c379)'
                        : 'var(--danger, #e06c75)',
                    }}>
                      {testResult.success ? '✓' : '✗'} {testResult.message}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
