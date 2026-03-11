/**
 * Rio Chat Page — Live conversation viewer with history
 *
 * Connects to the dashboard WebSocket for real-time transcript events,
 * fetches conversation history from the REST API, and renders a
 * ChatGPT-style interface with a sidebar conversation list.
 */

'use strict';

const RioChat = (() => {
  const API_BASE = window.location.origin;
  const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/dashboard`;

  let _ws = null;
  let _activeConvId = null;
  let _reconnectTimer = null;

  // DOM references
  const messagesEl = document.getElementById('chat-messages');
  const emptyEl = document.getElementById('chat-empty');
  const convListEl = document.getElementById('conversation-list');
  const btnNewChat = document.getElementById('btn-new-chat');
  const wsStatusEl = document.getElementById('ws-status');
  const wsLabelEl = document.getElementById('ws-status-label');

  // SVG icons
  const CHAT_ICON = `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
    <path d="M3 3h10a1 1 0 011 1v6a1 1 0 01-1 1H6l-3 3V4a1 1 0 011-1z" stroke="currentColor" stroke-width="1.3"/>
  </svg>`;

  const USER_AVATAR = `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
    <circle cx="8" cy="6" r="3" stroke="currentColor" stroke-width="1.4"/>
    <path d="M3 14c0-2.8 2.2-5 5-5s5 2.2 5 5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
  </svg>`;

  const RIO_AVATAR = `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
    <rect x="2" y="2" width="12" height="12" rx="3" stroke="currentColor" stroke-width="1.4"/>
    <circle cx="6" cy="7" r="1" fill="currentColor"/>
    <circle cx="10" cy="7" r="1" fill="currentColor"/>
    <path d="M6 10.5c.5.8 1.2 1 2 1s1.5-.2 2-1" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
  </svg>`;

  // ═══════════════════════════════════════════════════════════════════════
  // WebSocket connection (reuses dashboard WS for live transcript events)
  // ═══════════════════════════════════════════════════════════════════════

  function connectWS() {
    if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) return;

    _ws = new WebSocket(WS_URL);

    _ws.onopen = () => {
      _setStatus('connected');
      if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    };

    _ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === 'transcript') {
          _onLiveTranscript(data);
        }
      } catch { /* ignore non-JSON */ }
    };

    _ws.onclose = () => {
      _setStatus('disconnected');
      _scheduleReconnect();
    };

    _ws.onerror = () => {
      _setStatus('error');
    };
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) return;
    _reconnectTimer = setTimeout(() => {
      _reconnectTimer = null;
      connectWS();
    }, 3000);
  }

  function _setStatus(state) {
    const labels = { connected: 'Live', disconnected: 'Disconnected', error: 'Error', connecting: 'Connecting' };
    const colors = { connected: 'var(--success)', disconnected: 'var(--text-tertiary)', error: 'var(--error)', connecting: 'var(--warning)' };
    wsLabelEl.textContent = labels[state] || state;
    wsStatusEl.style.color = colors[state] || '';
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Conversation list
  // ═══════════════════════════════════════════════════════════════════════

  async function loadConversations() {
    try {
      const res = await fetch(`${API_BASE}/api/conversations`);
      if (!res.ok) return;
      const data = await res.json();
      _renderConversationList(data.conversations || []);
    } catch (err) {
      console.warn('Failed to load conversations:', err);
    }
  }

  function _renderConversationList(conversations) {
    if (!conversations.length) {
      convListEl.innerHTML = '<div class="chat-sidebar__empty">No conversations yet</div>';
      return;
    }

    convListEl.innerHTML = '';
    conversations.forEach(conv => {
      const el = document.createElement('div');
      el.className = 'conv-item' + (conv.id === _activeConvId ? ' conv-item--active' : '');
      el.dataset.convId = conv.id;

      const time = conv.created_at ? _formatDate(conv.created_at) : '';
      const preview = _escapeHtml(conv.preview || 'Conversation');

      el.innerHTML = `
        <div class="conv-item__icon">${CHAT_ICON}</div>
        ${conv.active ? '<div class="conv-item__live" title="Active"></div>' : ''}
        <div class="conv-item__body">
          <div class="conv-item__title">${preview || 'New conversation'}</div>
          <div class="conv-item__meta">
            <span class="conv-item__time">${time}</span>
            <span class="conv-item__count">${conv.message_count} msg</span>
          </div>
        </div>
      `;

      el.addEventListener('click', () => _selectConversation(conv.id));
      convListEl.appendChild(el);
    });
  }

  async function _selectConversation(convId) {
    _activeConvId = convId;
    // Highlight in sidebar
    convListEl.querySelectorAll('.conv-item').forEach(el => {
      el.classList.toggle('conv-item--active', el.dataset.convId === convId);
    });

    // Load messages
    try {
      const res = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(convId)}`);
      if (!res.ok) return;
      const data = await res.json();
      _renderMessages(data.messages || []);
    } catch (err) {
      console.warn('Failed to load conversation:', err);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Message rendering
  // ═══════════════════════════════════════════════════════════════════════

  function _renderMessages(messages) {
    // Clear
    messagesEl.innerHTML = '';
    if (emptyEl) emptyEl.style.display = 'none';

    if (!messages.length) {
      messagesEl.innerHTML = '';
      if (emptyEl) {
        messagesEl.appendChild(emptyEl);
        emptyEl.style.display = '';
      }
      return;
    }

    let lastDate = '';
    messages.forEach(msg => {
      // Day separator
      const msgDate = msg.timestamp ? new Date(msg.timestamp).toLocaleDateString() : '';
      if (msgDate && msgDate !== lastDate) {
        lastDate = msgDate;
        const sep = document.createElement('div');
        sep.className = 'chat-day-sep';
        sep.textContent = _formatDayLabel(msg.timestamp);
        messagesEl.appendChild(sep);
      }

      messagesEl.appendChild(_createMessageEl(msg));
    });

    // Scroll to bottom
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function _createMessageEl(msg) {
    const speaker = msg.speaker || 'system';
    const el = document.createElement('div');
    el.className = `chat-msg chat-msg--${speaker}`;

    const avatar = speaker === 'user' ? USER_AVATAR : RIO_AVATAR;
    const timeStr = msg.timestamp ? _formatTime(msg.timestamp) : '';
    const source = msg.source === 'transcription' ? '<span class="chat-msg__source">voice</span>' : '';

    el.innerHTML = `
      <div class="chat-msg__avatar">${avatar}</div>
      <div>
        <div class="chat-msg__bubble">${_escapeHtml(msg.text || '')}</div>
        <div class="chat-msg__time">${timeStr}${source}</div>
      </div>
    `;
    return el;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Live transcript handler
  // ═══════════════════════════════════════════════════════════════════════

  function _onLiveTranscript(data) {
    // Add to current view if we're viewing the active conversation or none
    const msg = {
      speaker: data.speaker || 'system',
      text: data.text || '',
      timestamp: new Date().toISOString(),
      source: data.source || '',
    };

    // Hide empty state
    if (emptyEl) emptyEl.style.display = 'none';

    messagesEl.appendChild(_createMessageEl(msg));

    // Auto-scroll if near bottom
    const threshold = 100;
    const isNearBottom = (messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight) < threshold;
    if (isNearBottom) {
      requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      });
    }

    // Refresh sidebar periodically
    _debouncedRefreshSidebar();
  }

  let _refreshTimeout = null;
  function _debouncedRefreshSidebar() {
    if (_refreshTimeout) return;
    _refreshTimeout = setTimeout(() => {
      _refreshTimeout = null;
      loadConversations();
    }, 5000);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // New conversation button
  // ═══════════════════════════════════════════════════════════════════════

  btnNewChat.addEventListener('click', async () => {
    try {
      const res = await fetch(`${API_BASE}/api/conversations/new`, { method: 'POST' });
      if (!res.ok) return;
      const data = await res.json();
      _activeConvId = data.id;
      _renderMessages([]);
      await loadConversations();
    } catch (err) {
      console.warn('Failed to start new conversation:', err);
    }
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Utilities
  // ═══════════════════════════════════════════════════════════════════════

  function _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function _formatTime(ts) {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function _formatDate(ts) {
    const d = new Date(ts);
    const now = new Date();
    const diff = Math.floor((now - d) / 86400000);
    if (diff === 0) return 'Today';
    if (diff === 1) return 'Yesterday';
    if (diff < 7) return d.toLocaleDateString([], { weekday: 'short' });
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  function _formatDayLabel(ts) {
    const d = new Date(ts);
    const now = new Date();
    const diff = Math.floor((now - d) / 86400000);
    if (diff === 0) return 'Today';
    if (diff === 1) return 'Yesterday';
    return d.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Init
  // ═══════════════════════════════════════════════════════════════════════

  async function init() {
    connectWS();
    await loadConversations();

    // Auto-select first conversation or load current chat history
    const convItems = convListEl.querySelectorAll('.conv-item');
    if (convItems.length > 0) {
      const firstId = convItems[0].dataset.convId;
      await _selectConversation(firstId);
    } else {
      // No saved conversations — load current live buffer
      try {
        const res = await fetch(`${API_BASE}/api/chat-history?limit=200`);
        if (res.ok) {
          const data = await res.json();
          if (data.messages && data.messages.length > 0) {
            _renderMessages(data.messages);
          }
        }
      } catch { /* ignore */ }
    }
  }

  init();

  return { loadConversations };
})();
