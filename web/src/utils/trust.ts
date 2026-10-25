import type { RequirementAcceptanceInfo, TrustInfo } from '../types';

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
  if (trust.completionStatus) return trust.completionStatus;
  if (trust.budgetExhausted) return 'incomplete';
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

function hasPageReadback(check?: RequirementAcceptanceInfo | null): boolean {
  return check?.readback?.status === 'pass' && check.readback.scope === 'observed_page_state'
    || check?.components?.some(hasPageReadback) === true;
}

function acknowledgedWrite(check?: RequirementAcceptanceInfo | null): boolean {
  return check?.status === 'acknowledged';
}

function resultReadback(check?: RequirementAcceptanceInfo | null): RequirementAcceptanceInfo['independentReadback'] {
  return check?.independentReadback ?? check?.components?.map(resultReadback).find(Boolean);
}

function describeCompletion(trust: TrustInfo): TrustPresentation {
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
  const fetched = resultReadback(trust.requirementAcceptance);
  if (fetched && fetched.status !== 'pass'
    && (completion === 'unknown' || completion === 'completed' || trust.reason === 'task_incomplete')) {
    return { ...base, title: 'Result unconfirmed', tone: 'neutral', glyph: '·', ok: null, canEscalate: false,
      note: fetched.detail || 'The separate result check could not confirm the fields. This does not establish that saving failed; review before repeating the action.' };
  }
  if (hasPageReadback(trust.requirementAcceptance)
    && (completion === 'unknown' || completion === 'incomplete' && trust.budgetExhausted)) {
    return {
      ...base, title: completion === 'incomplete' ? 'Page checked · run stopped' : 'Page checked · scope unconfirmed',
      note: 'The observed page matched the checks. This does not confirm every part of the request. Review the remaining work before repeating an action.',
      tone: 'neutral', glyph: '·', ok: null, canEscalate: false,
    };
  }
  if (verification === 'unavailable') {
    return {
      ...base, title: 'Verification unavailable', tone: 'warning', glyph: '⚠', ok: false,
      canEscalate: false,
      note: trust.honestNote || 'A required check could not run. The result is unconfirmed; no mismatch was established.',
    };
  }
  if (completion === 'unknown' && trust.reason === 'verification_inconclusive') {
    return {
      ...base, title: 'Result unconfirmed', tone: 'warning', glyph: '⚠', ok: false,
      canEscalate: false,
      note: trust.honestNote || 'Some requested results could not be confirmed. Review the evidence before repeating an action.',
    };
  }
  if (completion === 'incomplete') {
    if (verification === 'passed' && trust.reason === 'max_gate_blocked'
      && trust.completionCheck?.name === 'Test result claims'
      && trust.explanationReview?.status === 'unavailable') {
      return {
        ...base, title: 'Tests passed · summary unchecked', tone: 'warning', glyph: '⚠', ok: false,
        canEscalate: false,
        note: 'The recorded tests passed. The explanation review was unavailable; the full task is not marked complete.',
      };
    }
    if (trust.reason === 'desktop_blocked') {
      return {
        ...base, title: 'Desktop task stopped', tone: 'warning', glyph: '⚠', ok: false,
        canEscalate: false,
        note: trust.completionCheck?.detail || trust.honestNote || 'The desktop result could not be confirmed.',
      };
    }
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
  if (completion === 'completed' && acknowledgedWrite(trust.requirementAcceptance)) {
    return { ...base, title: 'Saved · server accepted', tone: 'neutral', glyph: '✓', ok: true, canEscalate: false,
      note: 'The site accepted the save and the submitted fields matched. The stored record was not read back.' };
  }
  if (required && verification !== 'passed') {
    return {
      ...base, title: verification === 'inconclusive' ? 'Verification inconclusive' : 'Verification required',
      note: trust.requirementAcceptance?.required && trust.requirementAcceptance.status !== 'pass'
        ? 'The result was delivered, but some requested conditions could not be confirmed.'
        : trust.tableAcceptance?.unverified.length
        ? 'Some requested data conditions could not be checked. See the details below.'
        : 'The latest changes still need a passing check.', tone: 'warning', glyph: '⚠', ok: false,
    };
  }
  if (completion === 'unknown') {
    return { ...base, title: 'Status unavailable', note: 'No completion status was reported.', tone: 'neutral', glyph: '·', ok: null, showCard: Boolean(trust.artifactReceipts?.length || trust.changeReceipts?.length) };
  }
  if (completion === 'completed' && verification === 'passed' && trust.explanationReview?.status === 'unavailable') {
    return {
      ...base, title: 'Completed · summary unchecked', tone: 'warning', glyph: '✓', ok: true,
      canEscalate: false, showCard: true,
      note: 'Execution checks passed. The explanation review was unavailable; its claims remain unverified.',
    };
  }
  if (verification === 'passed') {
    if (fetched?.status === 'pass') {
      return { ...base, title: 'Retrieved fields checked', tone: 'neutral', glyph: '✓', ok: true,
        note: 'The checked fields matched a separate response from the result URL. This check covers those fields only.'
          + (trust.budgetExhausted ? ' Further model requests stopped at the budget limit.' : '') };
    }
    if (hasPageReadback(trust.requirementAcceptance)) {
      return { ...base, title: 'Page checked',
        note: 'The page matched the requested checks. Storage in the app was not independently verified.',
        tone: 'neutral', glyph: '✓', ok: true };
    }
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
    tone: 'neutral', glyph: '✓', ok: true, showCard: Boolean(trust.artifactReceipts?.length || trust.changeReceipts?.length),
  };
}

/** The chat card, progress pane and status indicator use the same verdict. */
export function describeTrust(trust: TrustInfo): TrustPresentation {
  let presentation = describeCompletion(trust);
  if (completionStatus(trust) === 'completed' && trust.sourceReview?.excerptOnlyNumbers?.length) {
    presentation = { ...presentation, note: `${presentation.note} Some numbers appear only in search excerpts; their use in this answer was not checked.`,
      tone: 'warning', showCard: true };
  }
  const reserveNote = {
    budget: 'Further retrieval stopped to preserve the remaining request budget for this answer.',
    time: 'Further retrieval stopped to leave time for this answer before the run deadline.',
    tool_rounds: 'The tool-round limit was reached. This answer uses the sources already retrieved.',
    '': '',
  }[trust.answerReserveReason || ''];
  if (!reserveNote || ['cancelled', 'interrupted', 'failed'].includes(completionStatus(trust))) return presentation;
  if (presentation.title === 'Completed') {
    return { ...presentation, title: 'Response returned', note: `${presentation.note} ${reserveNote}`, showCard: true, ok: null, glyph: '·' };
  }
  return { ...presentation, note: `${presentation.note} ${reserveNote}`, showCard: true };
}

export function checkEvidence(trust: TrustInfo): string {
  if (trust.completionStatus === 'completed' && trust.explanationReview?.status === 'unavailable') {
    return trust.explanationReview.detail;
  }
  if (['completed_gate_warnings', 'max_gate_blocked', 'desktop_blocked',
    'verification_unavailable', 'verification_inconclusive', 'task_incomplete'].includes(trust.reason)
    && trust.completionCheck?.detail) {
    return `${trust.completionCheck.name}\n${trust.completionCheck.detail}`;
  }
  return trust.requirementAcceptance?.detail?.trim() || trust.evidenceGate?.lastEvidence?.trim() || '';
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
