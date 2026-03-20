/**
 * RUNE Browser Bridge — Popup UI
 *
 * 상태 표시 전용. 연결은 background.js가 자동으로 처리.
 */

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const waitingSection = document.getElementById('waitingSection');
const connectedSection = document.getElementById('connectedSection');
const disconnectBtn = document.getElementById('disconnectBtn');
const portInfo = document.getElementById('portInfo');
const errorInfo = document.getElementById('errorInfo');

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
  waitingSection.style.display = state === 'connected' ? 'none' : 'block';
  connectedSection.style.display = state === 'connected' ? 'block' : 'none';
}

function showError(msg) {
  if (errorInfo) {
    errorInfo.textContent = msg;
    errorInfo.style.display = msg ? 'block' : 'none';
  }
}

disconnectBtn.addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'disconnect' }, () => {
    setStatus('waiting', '자동 연결 대기 중...');
    showError('');
  });
});

// background → popup 상태 수신
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'status') {
    if (message.connected) {
      setStatus('connected', 'Connected');
      showError('');
    } else {
      setStatus('waiting', '자동 연결 대기 중...');
      if (message.error) showError(message.error);
    }
  }
});

// 초기 상태 로드
chrome.runtime.sendMessage({ type: 'getStatus' }, (response) => {
  if (!response) return;
  if (response.connected) {
    setStatus('connected', 'Connected');
  } else {
    setStatus('waiting', '자동 연결 대기 중...');
    if (response.error) showError(response.error);
  }
});
