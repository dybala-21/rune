import { post } from './api';

export interface AzureCheckResult {
  ok: boolean;
  detail: string;
  models?: string[];
  statusCode?: number;
  latencyMs?: number;
}

export interface AzureDraftConnection {
  endpoint: string;
  apiKey?: string;
}

export function fetchAzureModels(connection: AzureDraftConnection): Promise<AzureCheckResult> {
  return post('/api/v1/azure/models', connection);
}

export function testAzureConnection(connection: AzureDraftConnection, deployment?: string, model?: string): Promise<AzureCheckResult> {
  return post('/api/v1/azure/test', { ...connection, deployment, model });
}
