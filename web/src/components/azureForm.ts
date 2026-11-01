import type { ConfigInfo } from '../api';

export type AzureConnection = NonNullable<ConfigInfo['azure']>;
export type AzureRow = { id: string; name: string; model: string; pricing: string };
export const modelName = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const azureHosts = ['.openai.azure.com', '.services.ai.azure.com', '.cognitiveservices.azure.com',
  '.openai.azure.us', '.cognitiveservices.azure.us', '.openai.azure.cn', '.cognitiveservices.azure.cn'];

export function resource(value: string): string {
  return value.trim().replace(/\/+$/, '').replace(/\/openai\/v1$/, '');
}

export function connectionErrors(connection: AzureConnection, endpoint: string, key: string): Record<string, string> {
  const errors: Record<string, string> = {};
  try {
    const url = new URL(endpoint.trim());
    if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash
        || url.port || !azureHosts.some(host => url.hostname.endsWith(host))
        || !['', '/openai/v1'].includes(url.pathname.replace(/\/+$/, ''))) throw new Error();
  } catch { errors['azure-endpoint'] = 'Enter the HTTPS resource endpoint from Azure Foundry.'; }
  if (!key.trim() && (!connection.hasKey || resource(endpoint) !== resource(connection.endpoint))) {
    errors['azure-key'] = connection.keyScope === 'process'
      ? 'This key comes from the launch environment. Change it there and restart Rune to switch resources.'
      : 'Enter the API key for this resource.';
  } else if (key.trim() && !/^[\x20-\x7e]+$/.test(key.trim())) {
    errors['azure-key'] = 'Enter an API key without line breaks or non-ASCII characters.';
  }
  return errors;
}

export function rowErrors(rows: AzureRow[]): Record<string, string> {
  const errors: Record<string, string> = {};
  const seen = new Map<string, string>();
  for (const row of rows) {
    if (!modelName.test(row.name.trim())) errors[`${row.id}-name`] = 'Enter the exact deployment name: letters, numbers, dots, underscores or hyphens (up to 128 characters).';
    if (!modelName.test(row.model.trim())) errors[`${row.id}-model`] = 'Enter the underlying model ID, or choose it from the catalog.';
    const previous = seen.get(row.name.trim());
    if (previous) {
      errors[`${previous}-name`] = errors[`${row.id}-name`] = 'This deployment name is already used in another row.';
    }
    seen.set(row.name.trim(), row.id);
  }
  return errors;
}

export function focusError(errors: Record<string, string>) {
  requestAnimationFrame(() => {
    const input = [...document.querySelectorAll<HTMLElement>('.azure-settings input, .azure-settings select')]
      .find(element => element.id in errors);
    const target = input?.matches(':disabled') ? document.getElementById(`${input.id}-error`) : input;
    target?.focus(); target?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  });
}
