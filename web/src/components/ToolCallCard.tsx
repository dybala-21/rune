import { memo } from 'react';
import type { ToolCall } from '../types';
import { normalizeToolName } from '../utils/tooling';
import { SearchActivity } from './SearchActivity';
import { ToolActivity } from './ToolActivity';

export const ToolCallCard = memo(function ToolCallCard({ toolCall, active = false }: { toolCall: ToolCall; active?: boolean }) {
  if (!toolCall.toolName?.trim()) return null;
  return normalizeToolName(toolCall.toolName) === 'web.search'
    ? <SearchActivity tools={[toolCall]} active={active} />
    : <ToolActivity tools={[toolCall]} active={active} />;
});
