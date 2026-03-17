/* ==========================================================================
   Rio Dashboard — Tool Execution Log
   Shows live tool_call + tool_result events with SVG status icons.
   ========================================================================== */

'use strict';

const RioToolLog = (() => {
  const MAX_ENTRIES = 50;

  let _container = null;
  let _emptyEl = null;
  let _countEl = null;
  let _pendingToolsById = {}; // toolId -> DOM element (for matching results)
  let _pendingToolQueuesByName = {}; // toolName -> [toolId, ...] (FIFO for id-less results)
  let _toolCount = 0;

  // SVG icons for each tool
  const TOOL_ICONS = {
    read_file: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <path d="M4 2h6l4 4v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" stroke-width="1.3"/>
      <path d="M10 2v4h4" stroke="currentColor" stroke-width="1.3"/>
    </svg>`,
    write_file: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <path d="M4 2h6l4 4v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" stroke-width="1.3"/>
      <path d="M6 9h4M8 7v4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    </svg>`,
    patch_file: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <path d="M4 2h6l4 4v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" stroke-width="1.3"/>
      <path d="M6 8h4M6 11h2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    </svg>`,
    run_command: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" stroke-width="1.3"/>
      <path d="M5 7l2 2-2 2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M9 11h3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    </svg>`,
    capture_screen: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <rect x="2" y="3" width="12" height="10" rx="1" stroke="currentColor" stroke-width="1.3"/>
      <circle cx="8" cy="8" r="2" stroke="currentColor" stroke-width="1.3"/>
    </svg>`,
    _default: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none">
      <circle cx="8" cy="8" r="5" stroke="currentColor" stroke-width="1.3"/>
      <path d="M8 6v4M8 12h.01" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    </svg>`
  };

  // Status SVGs
  const STATUS_SVG = {
    pending: `<svg viewBox="0 0 12 12" width="10" height="10" fill="none">
      <circle cx="6" cy="6" r="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2"/>
    </svg>`,
    success: `<svg viewBox="0 0 12 12" width="10" height="10" fill="none">
      <path d="M3 6.5l2 2 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
    error: `<svg viewBox="0 0 12 12" width="10" height="10" fill="none">
      <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`
  };

  function init() {
    _container = document.getElementById('tools-body');
    _emptyEl = document.getElementById('tools-empty');
    _countEl = document.getElementById('tools-count');

    // Listen for tool events
    RioSocket.on('tool_call', (data) => {
      addToolCall(data.name, data.args || {}, data.tool_id || data.id || data.call_id);
    });

    RioSocket.on('tool_result', (data) => {
      resolveToolResult(data.tool_id || data.id || data.call_id || null, _parseSuccess(data), data.name);
    });

    RioSocket.on('tool_event_update', (data) => {
      resolveToolResult(data.tool_id || data.id || data.call_id || null, _parseSuccess(data), data.name);
    });
  }

  function addToolCall(name, args, toolId) {
    if (!_container) return;

    const safeName = (name || 'unknown').toString();
    const safeArgs = args && typeof args === 'object' ? args : {};

    // Hide empty state
    if (_emptyEl) _emptyEl.style.display = 'none';

    _toolCount++;
    if (_countEl) _countEl.textContent = _toolCount;

    // Use server-provided tool_id, or generate a unique local one
    const id = String(toolId || `local_${_toolCount}`);

    const toolIcon = TOOL_ICONS[safeName] || TOOL_ICONS._default;
    const detail = _formatArgs(safeName, safeArgs);
    const timeStr = _now();

    const el = document.createElement('div');
    el.className = 'tool-entry';
    el.dataset.toolName = safeName;
    el.dataset.toolId = id;
    el.innerHTML = `
      <div class="tool-entry__icon tool-entry__icon--pending">
        ${toolIcon}
      </div>
      <div class="tool-entry__body">
        <div class="tool-entry__name">${_escapeHtml(safeName)}</div>
        <div class="tool-entry__detail" title="${_escapeHtml(detail)}">${_escapeHtml(detail)}</div>
      </div>
      <span class="tool-entry__time">${timeStr}</span>
    `;

    _container.appendChild(el);

    // Track by both ID and name queue so we can resolve even if results do not include IDs.
    _pendingToolsById[id] = el;
    if (!_pendingToolQueuesByName[safeName]) _pendingToolQueuesByName[safeName] = [];
    _pendingToolQueuesByName[safeName].push(id);

    // Prune old
    while (_container.querySelectorAll('.tool-entry').length > MAX_ENTRIES) {
      const first = _container.querySelector('.tool-entry');
      if (first) {
        _dropPending(first.dataset.toolId, first.dataset.toolName);
        first.remove();
      }
    }

    // Scroll to bottom
    _container.scrollTop = _container.scrollHeight;
  }

  function resolveToolResult(toolId, success, toolName) {
    const resolvedId = _resolvePendingId(toolId, toolName);
    if (!resolvedId) return;

    const el = _pendingToolsById[resolvedId];
    if (!el) {
      _dropPending(resolvedId, toolName);
      return;
    }

    _dropPending(resolvedId, el.dataset.toolName);

    const name = el.dataset.toolName;
    const iconEl = el.querySelector('.tool-entry__icon');
    if (iconEl) {
      iconEl.classList.remove('tool-entry__icon--pending');
      if (success !== false) {
        iconEl.classList.add('tool-entry__icon--success');
        // Replace inner SVG with check
        const toolIcon = TOOL_ICONS[name] || TOOL_ICONS._default;
        iconEl.innerHTML = toolIcon;
      } else {
        iconEl.classList.add('tool-entry__icon--error');
        const toolIcon = TOOL_ICONS[name] || TOOL_ICONS._default;
        iconEl.innerHTML = toolIcon;
      }
    }
  }

  function _resolvePendingId(toolId, toolName) {
    const id = toolId ? String(toolId) : '';
    const name = (toolName || '').toString();

    if (id && _pendingToolsById[id]) {
      return id;
    }

    if (name) {
      const queue = _pendingToolQueuesByName[name];
      if (queue && queue.length) {
        while (queue.length) {
          const candidateId = queue[0];
          if (_pendingToolsById[candidateId]) {
            return candidateId;
          }
          queue.shift();
        }
      }
    }

    // Backwards-compat: some older frames send name in place of tool_id.
    if (id) {
      const queue = _pendingToolQueuesByName[id];
      if (queue && queue.length) {
        while (queue.length) {
          const candidateId = queue[0];
          if (_pendingToolsById[candidateId]) {
            return candidateId;
          }
          queue.shift();
        }
      }
    }

    return null;
  }

  function _dropPending(toolId, toolName) {
    const id = toolId ? String(toolId) : '';
    const name = (toolName || '').toString();
    if (id) delete _pendingToolsById[id];
    if (!name) return;

    const queue = _pendingToolQueuesByName[name];
    if (!queue) return;
    const idx = queue.indexOf(id);
    if (idx >= 0) queue.splice(idx, 1);
    if (queue.length === 0) delete _pendingToolQueuesByName[name];
  }

  function _parseSuccess(data) {
    if (!data || typeof data !== 'object') return true;
    if (typeof data.success === 'boolean') return data.success;
    if (data.result && typeof data.result.success === 'boolean') return data.result.success;
    return true;
  }

  function _formatArgs(name, args) {
    if (name === 'read_file' || name === 'write_file' || name === 'patch_file') {
      return args.path || '';
    }
    if (name === 'run_command') {
      return args.command || '';
    }
    if (name === 'capture_screen') {
      return 'Taking screenshot...';
    }
    // Generic
    const keys = Object.keys(args);
    if (keys.length === 0) return '';
    return keys.map(k => `${k}: ${_truncate(String(args[k]), 40)}`).join(', ');
  }

  function _truncate(str, max) {
    return str.length > max ? str.slice(0, max - 3) + '...' : str;
  }

  function _now() {
    const d = new Date();
    return d.getHours().toString().padStart(2, '0') + ':' +
           d.getMinutes().toString().padStart(2, '0') + ':' +
           d.getSeconds().toString().padStart(2, '0');
  }

  function _escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  function getCount() {
    return _toolCount;
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { addToolCall, resolveToolResult, getCount };
})();
