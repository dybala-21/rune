import type { TrustInfo } from '../types';

export interface TrustPresentation {
  title: string;
  note: string;
  tone: 'neutral' | 'success' | 'warning' | 'danger';
  glyph: string;
  ok: boolean | null;
  showCard: boolean;
  canEscalate: boolean;
}

function completionStatus(trust: TrustInfo): NonNullable<TrustInfo['completionStatus']> {
  if (trust.completionStatus === 'cancelled' || trust.reason === 'cancelled') return 'cancelled';
  if (trust.completionStatus === 'failed' || trust.reason === 'error' || trust.reason.startsWith('error:')) return 'failed';
  if (trust.budgetExhausted) return 'incomplete';
  if (trust.completionStatus) return trust.completionStatus;
  return trust.reason === 'completed' || trust.reason === 'verified'
    ? 'completed' : trust.reason ? 'incomplete' : 'unknown';
}

function verificationStatus(trust: TrustInfo): NonNullable<TrustInfo['verificationStatus']> {
  if (trust.verificationStatus) return trust.verificationStatus;
  // Older daemons report a boolean that also means "no check". Use the evidence.
  if (trust.verification?.status === 'fail' || trust.evidenceGate?.lastVerdict === 'fail') return 'failed';
  if (trust.verification?.required && trust.verification.status !== 'pass') return 'not_checked';
  if (trust.testsPassedAfterEdit === false && trust.verification?.status !== 'pass') return 'not_checked';
  if (trust.verification?.status === 'pass' || trust.evidenceGate?.lastVerdict === 'pass'
    || trust.testsPassedAfterEdit === true) return 'passed';
  return trust.evidenceGate?.hasCheck ? 'inconclusive' : 'not_checked';
}

/** The chat card, progress pane and status indicator use the same verdict. */
export function describeTrust(trust: TrustInfo): TrustPresentation {
  const completion = completionStatus(trust);
  const verification = verificationStatus(trust);
  const required = trust.verificationRequired ?? (Boolean(trust.verification?.required)
    || trust.testsPassedAfterEdit === false);
  const base = {
    canEscalate: trust.canEscalate === true && completion === 'incomplete',
    showCard: true,
  };
  if (completion === 'cancelled') {
    return { ...base, title: 'Stopped', note: 'This run was cancelled.', tone: 'neutral', glyph: '◼', ok: false };
  }
  if (completion === 'interrupted') {
    return { ...base, title: 'Interrupted', note: 'The server stopped before this run finished. Saved progress is shown; no tools were restarted.', tone: 'warning', glyph: '◼', ok: false };
  }
  if (completion === 'failed') {
    return { ...base, title: 'Run failed', note: trust.honestNote || 'An execution error interrupted this run.', tone: 'danger', glyph: '✗', ok: false };
  }
  if (verification === 'failed') {
    return { ...base, title: 'Checks failed', note: trust.honestNote || 'A recorded check failed. Review the evidence before using this result.', tone: 'warning', glyph: '⚠', ok: false };
  }
  if (completion === 'incomplete') {
    if (trust.reason === 'completed_gate_warnings' && !trust.budgetExhausted) {
      return {
        ...base, title: 'Review needed', tone: 'warning', glyph: '⚠', ok: false,
        note: trust.completionCheck?.detail
          ? `The response was returned, but this check remains unresolved: ${trust.completionCheck.name}.`
          : 'The response was returned, but completion checks remain unresolved. The specific check was not recorded.',
      };
    }
    return {
      ...base, title: 'Incomplete', tone: 'warning', glyph: '⚠', ok: false,
      note: trust.budgetExhausted
        ? 'The tool limit was reached; some requested work may be unfinished.'
        : trust.honestNote || 'This run stopped before completing the request.',
    };
  }
  if (required && verification !== 'passed') {
    return {
      ...base, title: verification === 'inconclusive' ? 'Verification inconclusive' : 'Verification required',
      note: trust.tableAcceptance?.unverified.length
        ? 'Some requested data conditions could not be checked. See the details below.'
        : 'The latest changes still need a passing check.', tone: 'warning', glyph: '⚠', ok: false,
    };
  }
  if (completion === 'unknown') {
    return { ...base, title: 'Status unavailable', note: 'No completion status was reported.', tone: 'neutral', glyph: '·', ok: null, showCard: Boolean(trust.artifactReceipts?.length) };
  }
  if (verification === 'passed') {
    if (trust.tableAcceptance?.status === 'pass' && !trust.testsPassedAfterEdit && !trust.evidenceGate?.hasCheck) {
      return { ...base, title: 'Table data checked',
        note: 'The saved table matches the interpreted data requirements. These checks do not cover prose or layout.',
        tone: 'success', glyph: '✓', ok: true };
    }
    const testsOnly = trust.testsPassedAfterEdit === true && !trust.evidenceGate?.hasCheck;
    return {
      ...base, title: testsOnly ? 'Tests passing' : 'Checks passed',
      note: testsOnly ? 'Tests passed after the latest code change.' : 'The recorded verification checks passed.',
      tone: 'success', glyph: '✓', ok: true,
    };
  }
  if (verification === 'inconclusive') {
    return { ...base, title: 'Verification inconclusive', note: 'The run completed, but its check did not produce a verdict.', tone: 'neutral', glyph: '·', ok: true };
  }
  return {
    ...base, title: 'Completed', note: 'Result verification was not performed.',
    tone: 'neutral', glyph: '✓', ok: true, showCard: Boolean(trust.artifactReceipts?.length),
  };
}

export function checkEvidence(trust: TrustInfo): string {
  if ((trust.reason === 'completed_gate_warnings' || trust.reason === 'max_gate_blocked')
    && trust.completionCheck?.detail) {
    return `${trust.completionCheck.name}\n${trust.completionCheck.detail}`;
  }
  return trust.evidenceGate?.lastEvidence?.trim() || '';
}

export function trustColors(tone: TrustPresentation['tone']): { accent: string; background: string } {
  if (tone === 'neutral') return { accent: 'var(--border)', background: 'var(--bg-secondary)' };
  return { accent: `var(--${tone})`, background: `var(--${tone}-subtle, var(--bg-secondary))` };
}

export function checkSummary(trust: TrustInfo): string {
  const gate = trust.evidenceGate;
  if (!gate?.hasCheck) return '';
  if (gate.lastVerdict === 'fail') return 'latest check failed';
  if (gate.lastVerdict !== 'pass') return 'no verdict';
  const count = gate.verdictCounts.pass ?? gate.verdictCounts.passed ?? 0;
  return count > 0 ? `${count} check${count > 1 ? 's' : ''} passed` : 'check passed';
}
