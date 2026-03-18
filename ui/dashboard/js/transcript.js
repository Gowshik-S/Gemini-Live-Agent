/* ==========================================================================
   Rio Dashboard — Live Transcript Module
   Renders user + Rio messages in the transcript panel.
   ========================================================================== */

'use strict';

const RioTranscript = (() => {
  const MAX_MESSAGES = 200; // Auto-prune oldest beyond this
  const LANGUAGE_OPTIONS = [
    { code: 'auto', label: 'Auto detect' },
    { code: 'en', label: 'English' },
    { code: 'ta', label: 'Tamil' },
    { code: 'es', label: 'Spanish' },
    { code: 'fr', label: 'French' },
    { code: 'de', label: 'German' },
    { code: 'it', label: 'Italian' },
    { code: 'pt', label: 'Portuguese' },
    { code: 'hi', label: 'Hindi' },
    { code: 'ar', label: 'Arabic' },
    { code: 'ja', label: 'Japanese' },
    { code: 'ko', label: 'Korean' },
    { code: 'zh', label: 'Chinese' },
    { code: 'ru', label: 'Russian' },
  ];

  let _container = null;
  let _emptyEl = null;
  let _countEl = null;
  let _translatorEnabledEl = null;
  let _translatorBidiEl = null;
  let _translatorStaffLangEl = null;
  let _translatorCounterpartyLangEl = null;
  let _translatorStatusEl = null;
  let _messageCount = 0;
  let _autoScroll = true;
  let _suppressTranslatorChange = false;

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
    translation: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none">
      <path d="M3 5h6M6 3v2M7 8l2-4 2 4M7.8 6.4h2.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
      <path d="M10.5 10.5h3M12 9v4M9.8 12h4.4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
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
    _translatorEnabledEl = document.getElementById('translator-enabled');
    _translatorBidiEl = document.getElementById('translator-bidirectional');
    _translatorStaffLangEl = document.getElementById('translator-staff-language');
    _translatorCounterpartyLangEl = document.getElementById('translator-counterparty-language');
    _translatorStatusEl = document.getElementById('translator-status');

    _initTranslatorControls();

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

    RioSocket.on('translation', (data) => {
      addTranslationMessage(data || {});
    });

    RioSocket.on('control', (data) => {
      if (data && data.action === 'translator_mode') {
        _syncTranslatorControls(data);
      }
    });

    // Fetch chat history on connect
    RioSocket.on('open', () => {
      _fetchChatHistory();
      _requestTranslatorStatus();
    });
  }

  function _initTranslatorControls() {
    if (!_translatorEnabledEl || !_translatorBidiEl || !_translatorStaffLangEl || !_translatorCounterpartyLangEl) {
      return;
    }

    _populateLanguageSelect(_translatorStaffLangEl, 'en');
    _populateLanguageSelect(_translatorCounterpartyLangEl, 'auto');

    _translatorEnabledEl.addEventListener('change', _onTranslatorControlChanged);
    _translatorBidiEl.addEventListener('change', _onTranslatorControlChanged);
    _translatorStaffLangEl.addEventListener('change', _onTranslatorControlChanged);
    _translatorCounterpartyLangEl.addEventListener('change', _onTranslatorControlChanged);

    _setTranslatorStatus(false, 'auto', 'en', false);
  }

  function _populateLanguageSelect(selectEl, selectedCode) {
    selectEl.innerHTML = '';
    LANGUAGE_OPTIONS.forEach((item) => {
      const opt = document.createElement('option');
      opt.value = item.code;
      opt.textContent = `${item.label} (${item.code})`;
      if (item.code === selectedCode) opt.selected = true;
      selectEl.appendChild(opt);
    });
  }

  function _requestTranslatorStatus() {
    RioSocket.send({ type: 'control', action: 'translator_status' });
  }

  function _onTranslatorControlChanged() {
    if (_suppressTranslatorChange) return;
    if (!_translatorEnabledEl || !_translatorBidiEl || !_translatorStaffLangEl || !_translatorCounterpartyLangEl) return;

    const enabled = Boolean(_translatorEnabledEl.checked);
    const bidirectional = Boolean(_translatorBidiEl.checked);
    const staffLanguage = _translatorStaffLangEl.value || 'en';
    const counterpartyLanguage = _translatorCounterpartyLangEl.value || 'auto';

    RioSocket.send({
      type: 'control',
      action: 'translator_update',
      payload: {
        enabled,
        bidirectional,
        staff_language: staffLanguage,
        counterparty_language: counterpartyLanguage,
        source_language: counterpartyLanguage,
        target_language: staffLanguage,
      },
    });

    _setTranslatorStatus(enabled, counterpartyLanguage, staffLanguage, bidirectional);
  }

  function _syncTranslatorControls(payload) {
    if (!_translatorEnabledEl || !_translatorBidiEl || !_translatorStaffLangEl || !_translatorCounterpartyLangEl) return;

    const enabled = Boolean(payload.enabled);
    const bidirectional = Boolean(payload.bidirectional);
    const sourceLanguage = String(payload.source_language || 'auto');
    const targetLanguage = String(payload.target_language || 'en');
    const staffLanguage = String(payload.staff_language || targetLanguage || 'en');
    const counterpartyLanguage = String(payload.counterparty_language || sourceLanguage || 'auto');

    _suppressTranslatorChange = true;
    _translatorEnabledEl.checked = enabled;
    _translatorBidiEl.checked = bidirectional;
    _translatorStaffLangEl.value = staffLanguage;
    _translatorCounterpartyLangEl.value = counterpartyLanguage;
    _suppressTranslatorChange = false;

    _setTranslatorStatus(enabled, counterpartyLanguage, staffLanguage, bidirectional);
  }

  function _setTranslatorStatus(enabled, sourceLanguage, targetLanguage, bidirectional) {
    if (!_translatorStatusEl) return;
    const sourceLabel = _languageLabel(sourceLanguage);
    const targetLabel = _languageLabel(targetLanguage);
    const mode = bidirectional ? 'bidirectional' : 'counterparty -> staff';
    _translatorStatusEl.textContent = enabled
      ? `Translator on: ${sourceLabel} -> ${targetLabel} (${mode})`
      : 'Translator off';
  }

  function _languageLabel(code) {
    const normalized = String(code || 'auto').toLowerCase();
    const found = LANGUAGE_OPTIONS.find((item) => item.code === normalized);
    return found ? found.label : normalized;
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

    const avatarType = speaker === 'user'
      ? 'user'
      : speaker === 'rio'
        ? 'rio'
        : speaker === 'translation'
          ? 'translation'
          : 'system';
    const speakerName = speaker === 'user'
      ? 'You'
      : speaker === 'rio'
        ? 'Rio'
        : speaker === 'translation'
          ? 'Translator'
          : 'System';
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

  function addTranslationMessage(payload) {
    const translatedText = payload.translated_text || payload.text || '';
    if (!translatedText) return;

    const direction = payload.direction || 'translation';
    const sourceLanguage = payload.source_language || 'auto';
    const targetLanguage = payload.target_language || 'en';
    const header = `${direction.replaceAll('_', ' ')} | ${sourceLanguage} -> ${targetLanguage}`;

    addMessage('translation', `${translatedText}\n\n(${header})`, payload.timestamp);
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

  return { addMessage, addTranslationMessage, getCount };
})();
