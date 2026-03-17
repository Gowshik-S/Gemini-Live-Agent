/* ==========================================================================
   Rio Dashboard — WebSocket Connection Manager
   Connects to /ws/dashboard, dispatches events to other modules.
   ========================================================================== */

'use strict';

const RioSocket = (() => {
  // ---- State ----
  let _ws = null;
  let _connected = false;
  let _reconnectTimer = null;
  let _reconnectDelay = 1000;
  let _pingTimer = null;
  let _connectTimeout = null;
  let _lastPong = 0;
  let _connectTime = 0;
  let _pingSentAt = 0;
  let _latencyMs = null;

  const MAX_RECONNECT_DELAY = 15000;
  const CONNECT_TIMEOUT_MS = 5000;

  // Event handlers registry
  const _handlers = {};

  // ---- Public API ----

  /**
   * Register an event handler.
   * Events: 'open', 'close', 'message', 'transcript', 'tool_call',
    *         'tool_result', 'tool_event_update', 'struggle', 'vision',
    *         'control', 'health', 'rate_limit'
   */
  function on(event, fn) {
    if (!_handlers[event]) _handlers[event] = [];
    _handlers[event].push(fn);
  }

  function _emit(event, data) {
    const fns = _handlers[event];
    if (fns) fns.forEach(fn => { try { fn(data); } catch (e) { console.error(`[rio-ws] handler error (${event}):`, e); } });
  }

  /** Connect to the dashboard WebSocket. */
  function connect() {
    if (_ws && (_ws.readyState === WebSocket.CONNECTING || _ws.readyState === WebSocket.OPEN)) {
      return; // Already connected / connecting
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8080';
    const url = `${protocol}//${host}/ws/dashboard`;

    _updateConnectionUI('connecting');
    console.log(`[rio-ws] connecting to ${url}`);

    try {
      _ws = new WebSocket(url);
    } catch (e) {
      console.error('[rio-ws] WebSocket constructor failed:', e);
      _scheduleReconnect();
      return;
    }

    // Guard against hanging CONNECTING state
    _startConnectTimeout();

    _ws.onopen = () => {
      _clearConnectTimeout();
      _connected = true;
      _reconnectDelay = 1000;
      _connectTime = Date.now();
      _lastPong = Date.now();
      console.log('[rio-ws] connected');
      _updateConnectionUI('connected');
      // Flush queued messages
      while (_sendQueue.length > 0) {
        _ws.send(_sendQueue.shift());
      }
      _emit('open', {});
      _startPing();
    };

    _ws.onmessage = (event) => {
      _lastPong = Date.now();

      // Track RTT for ping/pong latency measurement
      if (event.data === 'pong' && _pingSentAt > 0) {
        _latencyMs = Date.now() - _pingSentAt;
        _pingSentAt = 0;
        return; // Don't try to JSON-parse 'pong'
      }

      // Try JSON parse
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        console.warn('[rio-ws] non-JSON message:', event.data);
        return;
      }

      _emit('message', data);

      _routeMessage(data);
    };

    _ws.onerror = (e) => {
      console.error('[rio-ws] error:', e);
    };

    _ws.onclose = (e) => {
      _clearConnectTimeout();
      _connected = false;
      _stopPing();
      console.log(`[rio-ws] closed: code=${e.code} reason=${e.reason}`);
      _updateConnectionUI('disconnected');
      _emit('close', { code: e.code, reason: e.reason });
      _scheduleReconnect();
    };
  }

  /** Close the connection. */
  function disconnect() {
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
    _clearConnectTimeout();
    _stopPing();
    if (_ws) {
      _ws.close();
      _ws = null;
    }
    _connected = false;
  }

  // Queue for messages sent while connecting
  const _sendQueue = [];

  /** Send a text message. Queues if connecting, drops if disconnected. */
  function send(msg) {
    const payload = typeof msg === 'string' ? msg : JSON.stringify(msg);
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(payload);
    } else if (_ws && _ws.readyState === WebSocket.CONNECTING) {
      if (_sendQueue.length < 50) _sendQueue.push(payload);
    }
  }

  function isConnected() {
    return _connected;
  }

  function getLatency() {
    if (!_connected) return null;
    return _latencyMs;
  }

  function getConnectTime() {
    return _connectTime;
  }

  // ---- Internals ----

  function _routeMessage(rawData) {
    const envelope = _normalizeEnvelope(rawData);
    const type = envelope.type;
    const subtype = envelope.subtype;

    if (type === 'transcript' || (subtype === 'transcript' && type === 'dashboard')) {
      _emit('transcript', envelope.payload);
      return;
    }

    if (type === 'tool_call' || subtype === 'tool_call') {
      _emit('tool_call', _normalizeToolPayload(envelope.payload));
      return;
    }

    if (type === 'tool_result' || subtype === 'tool_result') {
      _emit('tool_result', _normalizeToolPayload(envelope.payload));
      return;
    }

    if (type === 'tool_event_update' || subtype === 'tool_event_update') {
      _emit('tool_event_update', _normalizeToolPayload(envelope.payload));
      return;
    }

    if (type === 'dashboard') {
      if (subtype === 'struggle') _emit('struggle', envelope.payload);
      if (subtype === 'vision') _emit('vision', envelope.payload);
      if (subtype === 'health') _emit('health', envelope.payload);
      if (subtype === 'rate_limit') _emit('rate_limit', envelope.payload);
      return;
    }

    if (type === 'control' || subtype === 'control') {
      _emit('control', envelope.payload);
      return;
    }
  }

  function _normalizeEnvelope(data) {
    const payload = data && typeof data === 'object' ? data : {};
    const nestedPayload = _firstObject(payload.payload, payload.data, payload.event, payload.detail);
    const merged = nestedPayload ? Object.assign({}, nestedPayload, payload) : payload;

    let type = _asString(
      merged.type,
      merged.event_type,
      merged.eventType,
      merged.kind,
      merged.channel,
    );

    let subtype = _asString(
      merged.subtype,
      merged.event,
      merged.event_name,
      merged.action,
      merged.name,
    );

    if (!type && subtype) type = 'dashboard';

    // Preserve existing transcript behavior when server sends speaker/text payload only.
    if (!type && merged.speaker && Object.prototype.hasOwnProperty.call(merged, 'text')) {
      type = 'transcript';
      subtype = 'transcript';
    }

    return {
      type,
      subtype,
      payload: merged,
    };
  }

  function _normalizeToolPayload(data) {
    const payload = data && typeof data === 'object' ? data : {};
    const result = _firstObject(payload.result, payload.payload) || {};

    const normalized = Object.assign({}, payload, {
      tool_id: payload.tool_id || payload.id || payload.call_id || null,
      name: payload.name || payload.tool_name || payload.tool || 'unknown',
      args: _firstObject(payload.args, payload.tool_args) || {},
    });

    if (typeof normalized.success !== 'boolean' && typeof result.success === 'boolean') {
      normalized.success = result.success;
    }

    return normalized;
  }

  function _asString(...values) {
    for (const value of values) {
      if (typeof value === 'string' && value.trim()) {
        return value.trim();
      }
    }
    return '';
  }

  function _firstObject(...values) {
    for (const value of values) {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        return value;
      }
    }
    return null;
  }

  function _startConnectTimeout() {
    _clearConnectTimeout();
    _connectTimeout = setTimeout(() => {
      if (_ws && _ws.readyState === WebSocket.CONNECTING) {
        console.warn('[rio-ws] connect timeout — aborting stale socket');
        _ws.close();
        _ws = null;
        _updateConnectionUI('disconnected');
        _scheduleReconnect();
      }
    }, CONNECT_TIMEOUT_MS);
  }

  function _clearConnectTimeout() {
    if (_connectTimeout) {
      clearTimeout(_connectTimeout);
      _connectTimeout = null;
    }
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) return;
    console.log(`[rio-ws] reconnecting in ${_reconnectDelay}ms`);
    _updateConnectionUI('connecting');
    _reconnectTimer = setTimeout(() => {
      _reconnectTimer = null;
      connect();
    }, _reconnectDelay);
    _reconnectDelay = Math.min(_reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
  }

  function _startPing() {
    _stopPing();
    _pingTimer = setInterval(() => {
      _pingSentAt = Date.now();
      send('ping');
    }, 10000);
  }

  function _stopPing() {
    if (_pingTimer) {
      clearInterval(_pingTimer);
      _pingTimer = null;
    }
  }

  function _updateConnectionUI(state) {
    const pill = document.getElementById('connection-pill');
    const label = document.getElementById('connection-label');
    if (!pill || !label) return;

    pill.className = 'status-pill';

    if (state === 'connected') {
      pill.classList.add('status-pill--connected');
      label.textContent = 'Connected';
    } else if (state === 'disconnected') {
      pill.classList.add('status-pill--disconnected');
      label.textContent = 'Disconnected';
    } else {
      pill.classList.add('status-pill--connecting');
      label.textContent = 'Connecting';
    }
  }

  // ---- Auto-connect on load ----
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    connect();
  }

  return { on, connect, disconnect, send, isConnected, getLatency, getConnectTime };
})();
