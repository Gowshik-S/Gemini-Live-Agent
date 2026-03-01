/* ==========================================================================
   Rio Dashboard — Live Transcript Module
   Renders user + Rio messages in the transcript panel.
   ========================================================================== */

'use strict';

const RioTranscript = (() => {
  const MAX_MESSAGES = 200; // Auto-prune oldest beyond this

  let _container = null;
  let _emptyEl = null;
  let _countEl = null;
  let _messageCount = 0;
  let _autoScroll = true;

  // SVG templates for avatars
  const AVATAR_SVG = {
    user: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <circle cx="8" cy="6" r="3" stroke="currentColor" stroke-width="1.4"/>
      <path d="M3 14c0-2.8 2.2-5 5-5s5 2.2 5 5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
    </svg>`,
    rio: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <rect x="2" y="2" width="12" height="12" rx="3" stroke="currentColor" stroke-width="1.4"/>
      <circle cx="6" cy="7" r="1" fill="currentColor"/>
      <circle cx="10" cy="7" r="1" fill="currentColor"/>
      <path d="M6 10.5c.5.8 1.2 1 2 1s1.5-.2 2-1" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
    </svg>`,
    system: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.4"/>
      <path d="M8 5v3l2 2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
    </svg>`
  };

  function init() {
    _container = document.getElementById('transcript-body');
    _emptyEl = document.getElementById('transcript-empty');
    _countEl = document.getElementById('transcript-count');

    // Track scroll position
    if (_container) {
      _container.addEventListener('scroll', () => {
        const threshold = 60;
        _autoScroll = (_container.scrollHeight - _container.scrollTop - _container.clientHeight) < threshold;
      });
    }

    // Listen for transcript events
    RioSocket.on('transcript', (data) => {
      addMessage(data.speaker || 'system', data.text || '', data.timestamp);
    });

    // Fetch chat history on connect
    RioSocket.on('open', _fetchChatHistory);
  }

  /** Fetch past messages from /api/chat-history on reconnect. */
  async function _fetchChatHistory() {
    try {
      const resp = await fetch('/api/chat-history?limit=100');
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(msg => {
          addMessage(msg.speaker || 'system', msg.text || '', msg.timestamp);
        });
      }
    } catch (e) {
      console.warn('[rio-transcript] failed to fetch chat history:', e);
    }
  }

  function addMessage(speaker, text, timestamp) {
    if (!_container || !text) return;

    // Hide empty state
    if (_emptyEl) {
      _emptyEl.style.display = 'none';
    }

    // Build message element
    const el = document.createElement('div');
    el.className = `msg msg--${speaker}`;

    const avatarType = speaker === 'user' ? 'user' : speaker === 'rio' ? 'rio' : 'system';
    const avatarLabel = speaker === 'user' ? 'You' : speaker === 'rio' ? 'R' : 'Sys';
    const speakerName = speaker === 'user' ? 'You' : speaker === 'rio' ? 'Rio' : 'System';
    const timeStr = _formatTime(timestamp);

    el.innerHTML = `
      <div class="msg__avatar" title="${speakerName}">
        ${AVATAR_SVG[avatarType]}
      </div>
      <div class="msg__content">
        <div class="msg__speaker">${speakerName}</div>
        <div class="msg__text">${_escapeHtml(text)}</div>
      </div>
      <span class="msg__time">${timeStr}</span>
    `;

    _container.appendChild(el);
    _messageCount++;

    // Update counter
    if (_countEl) {
      _countEl.textContent = _messageCount;
    }

    // Prune if needed
    while (_container.children.length > MAX_MESSAGES + 1) { // +1 for empty state div
      const first = _container.querySelector('.msg');
      if (first) first.remove();
    }

    // Auto-scroll
    if (_autoScroll) {
      requestAnimationFrame(() => {
        _container.scrollTop = _container.scrollHeight;
      });
    }
  }

  function _formatTime(ts) {
    const d = ts ? new Date(ts) : new Date();
    const h = d.getHours().toString().padStart(2, '0');
    const m = d.getMinutes().toString().padStart(2, '0');
    const s = d.getSeconds().toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  }

  function _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function getCount() {
    return _messageCount;
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { addMessage, getCount };
})();
