import type { useAgent } from '../hooks/useAgent';
import type { SessionHistoryState } from '../hooks/useSessionHistory';
import type { WorkbenchState } from '../hooks/useWorkbenchState';
import { RetainedPane } from './RetainedPane';
import { WorkbenchPanel } from './WorkbenchPanel';

interface Props {
  agent: ReturnType<typeof useAgent>;
  live: WorkbenchState;
  history: WorkbenchState;
  historyState: SessionHistoryState | null;
}

export function ConversationWorkbenches({ agent, live, history, historyState }: Props) {
  const viewingHistory = history.sessionId !== null;
  const shown = viewingHistory ? history : live;
  const close = () => { shown.setOpen(false); shown.setDismissed(true); };
  return (
    <div className="workbench-host" hidden={!shown.open}>
      <button className="back-to-chat" onClick={close}>
        ← Back to chat{!viewingHistory && (agent.pendingApproval || agent.pendingQuestion) ? ' · Needs your attention' : ''}
      </button>
      {/* Viewing history hides the live panel without ending its shell or browser connection. */}
      <RetainedPane key={live.sessionId} active={!viewingHistory && live.open}>
        <WorkbenchPanel
          sessionId={live.sessionId!}
          active={!viewingHistory && live.open}
          computerRequest={live.computerRequest}
          onTabChange={live.setTab}
          expanded={live.expanded}
          onToggleExpand={() => live.setExpanded(value => !value)}
          toolCalls={agent.toolCalls}
          fileChanges={agent.fileChanges}
          isRunning={agent.state === 'running'}
          activitySummary={agent.activitySummary}
          trust={agent.lastTrust}
          currentStep={agent.currentStepInfo}
          orchestration={agent.orchestration}
          awaiting={agent.state === 'waiting_approval' ? 'approval' : agent.state === 'waiting_question' ? 'question' : null}
          connected={agent.connected}
          onClose={close}
        />
      </RetainedPane>
      {viewingHistory && (
        <RetainedPane key={`history:${history.sessionId}`} active={history.open}>
          <WorkbenchPanel
            sessionId={history.sessionId!}
            historical
            active={history.open}
            onTabChange={history.setTab}
            toolCalls={historyState?.toolCalls ?? []}
            fileChanges={historyState?.run?.fileChanges ?? []}
            isRunning={false}
            activitySummary={historyState?.activitySummary ?? null}
            trust={historyState?.trust}
            connected={agent.connected}
            onClose={close}
          />
        </RetainedPane>
      )}
    </div>
  );
}
