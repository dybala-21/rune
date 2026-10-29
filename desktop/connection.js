'use strict';

const retry = document.getElementById('retry');
const status = document.getElementById('status');
retry.addEventListener('click', async () => {
  retry.disabled = true;
  status.textContent = 'Connecting to the local engine…';
  try {
    const result = await window.rune.retryConnection();
    if (result.error) status.textContent = result.error;
  } catch {
    status.textContent = 'Could not reconnect. Reopen Rune to try again.';
  } finally { retry.disabled = false; }
});
