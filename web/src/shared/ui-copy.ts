export const APPROVAL_COPY = {
  title: 'Approval Required',
  commandLabel: 'Command',
  reasonLabel: 'Reason',
  suggestionsLabel: 'Suggestions',
  actionsLabel: 'Select an action',
  allowOnceLabel: 'Allow Once',
  alwaysAllowLabel: 'Always Allow',
  denyLabel: 'Deny',
  denyWithInstructionsLabel: 'Deny with Instructions',
  instructionsLabel: 'Instructions',
  instructionsEmptyLabel: '(type here)',
  denialPlaceholder: 'Add denial instructions...',
  denialHintEmpty: 'Please type denial instructions first.',
  denialHintPrompt: 'Type denial instructions and press Enter.',
} as const;

export const QUESTION_COPY = {
  title: 'Question',
  answeredLabel: 'Answered',
  customOptionLabel: 'Custom input (guide the flow)',
  customOptionDescription: 'Type your own instruction to steer the next step.',
  answerNeededHeadline: 'Your answer is needed to proceed.',
  selectOrTypeHint: 'Use ↑/↓ + Enter to select, or type a custom answer.',
  customInputHint: 'Type your instructions and press Enter.',
  freeTextHint: 'Type your answer and press Enter.',
  selectOrTypePlaceholder: 'Use Up/Down + Enter to select, or type your own input',
  customInputPlaceholder: 'Enter your own instruction to steer the flow...',
  freeTextPlaceholder: 'Type your answer and press Enter...',
  freeTextFieldPlaceholder: 'Type your answer...',
  secretFieldPlaceholder: 'Enter secret value...',
  submitLabel: 'Submit',
} as const;

export function getQuestionEntryPlaceholder(
  hasOptions: boolean,
  isCustomSelected: boolean
): string {
  if (hasOptions) {
    return isCustomSelected
      ? QUESTION_COPY.customInputPlaceholder
      : QUESTION_COPY.selectOrTypePlaceholder;
  }
  return QUESTION_COPY.freeTextPlaceholder;
}

export function getQuestionGuidanceText(
  hasOptions: boolean,
  isCustomSelected: boolean
): string {
  if (hasOptions) {
    return isCustomSelected
      ? QUESTION_COPY.customInputHint
      : QUESTION_COPY.selectOrTypeHint;
  }
  return QUESTION_COPY.freeTextHint;
}

export function getQuestionFieldPlaceholder(inputMode: 'text' | 'secret' = 'text'): string {
  return inputMode === 'secret'
    ? QUESTION_COPY.secretFieldPlaceholder
    : QUESTION_COPY.freeTextFieldPlaceholder;
}
