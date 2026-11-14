import type { ConfigInfo, ModelConnection } from '../api';

/** Ignore stale setup errors after a model change. */
export function getModelSetupConnection(
  config: Pick<ConfigInfo, 'activeModel' | 'modelConnection'> | null | undefined,
): ModelConnection | null {
  const connection = config?.modelConnection;
  return connection?.state === 'needs_setup'
    && connection.provider === config?.activeModel.provider
    && connection.model === config.activeModel.model ? connection : null;
}
