import { post, type AzurePricing } from './api';

export interface AzureCheckResult {
  state?: 'cancelled' | 'incomplete';
  ok: boolean;
  detail: string;
  models?: string[];
  statusCode?: number;
  latencyMs?: number;
}

export interface AzureDraftConnection {
  endpoint: string;
  apiKey?: string;
  expectedRevision?: string;
}

export function fetchAzureModels(connection: AzureDraftConnection, signal?: AbortSignal): Promise<AzureCheckResult> {
  return post('/api/v1/azure/models', connection, false, signal);
}

export function testAzureConnection(connection: AzureDraftConnection, deployment?: string, model?: string, signal?: AbortSignal): Promise<AzureCheckResult> {
  return post('/api/v1/azure/test', { ...connection, deployment, model }, false, signal);
}

export function saveAzureSettings(settings: AzureDraftConnection & {
  revision: string; deployments: Record<string, string>; pricing: AzurePricing | null; useDeployment?: string;
}): Promise<{ saved: boolean; revision: string; credentialRevision: string; activeDeployment: string | null }> {
  return post('/api/v1/azure/settings', settings);
}
