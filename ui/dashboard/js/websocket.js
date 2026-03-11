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
   *         'tool_result', 'struggle', 'vision', 'control', 'health'
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

      // Route by type
      const type = data.type;
      if (type === 'transcript') {
        _emit('transcript', data);
      } else if (type === 'dashboard') {
        const sub = data.subtype;
        if (sub === 'tool_call')    _emit('tool_call', data);
        if (sub === 'tool_result')  _emit('tool_result', data);
        if (sub === 'struggle')     _emit('struggle', data);
        if (sub === 'vision')       _emit('vision', data);
        if (sub === 'health')       _emit('health', data);
        if (sub === 'rate_limit')   _emit('rate_limit', data);
      } else if (type === 'control') {
        _emit('control', data);
      }
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

  /** Send a text message (for keep-alive pings). */
  function send(msg) {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
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
