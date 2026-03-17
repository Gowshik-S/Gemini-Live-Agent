/* ==========================================================================
   Rio Dashboard — System Health Module
   Tracks connection status, latency, RPM, uptime, counters.
   ========================================================================== */

'use strict';

const RioHealth = (() => {
  // DOM refs
  let _latencyLabel = null;
  let _modelLabel = null;
  let _sessionLabel = null;
  let _rpmCurrent = null;
  let _rpmBar = null;
  let _uptimeValue = null;
  let _messagesValue = null;
  let _screenshotsValue = null;
  let _strugglesValue = null;
  let _ratelimitValue = null;
  let _ratelimitCard = null;

  // State
  let _uptimeStart = 0;
  let _uptimeTimer = null;
  let _latencyTimer = null;
  let _messageCount = 0;
  let _screenshotCount = 0;
  let _struggleCount = 0;
  let _rpm = 0;

  const RPM_BUDGET = 30;

  function init() {
    _latencyLabel   = document.getElementById('latency-label');
    _modelLabel     = document.getElementById('model-label');
    _sessionLabel   = document.getElementById('session-label');
    _rpmCurrent     = document.getElementById('rpm-current');
    _rpmBar         = document.getElementById('rpm-bar');
    _uptimeValue    = document.getElementById('uptime-value');
    _messagesValue  = document.getElementById('messages-value');
    _screenshotsValue = document.getElementById('screenshots-value');
    _strugglesValue = document.getElementById('struggles-value');
    _ratelimitValue = document.getElementById('ratelimit-value');
    _ratelimitCard  = document.getElementById('ratelimit-card');

    // Event listeners
    RioSocket.on('open', _onConnect);
    RioSocket.on('close', _onDisconnect);
    RioSocket.on('transcript', _onTranscript);
    RioSocket.on('vision', _onVision);
    RioSocket.on('struggle', _onStruggle);
    RioSocket.on('control', _onControl);
    RioSocket.on('health', _onHealth);
    RioSocket.on('rate_limit', _onRateLimit);
  }

  // ---- Event Handlers ----

  function _onConnect() {
    _uptimeStart = Date.now();
    _startUptime();
    _startLatencyPoll();
    _setLatencyPillState('muted');
  }

  function _onDisconnect() {
    _stopUptime();
    _stopLatencyPoll();
    if (_latencyLabel) _latencyLabel.textContent = '-- ms';
    _setLatencyPillState('muted');
  }

  function _onTranscript(data) {
    _messageCount++;
    if (_messagesValue) _messagesValue.textContent = _messageCount;
  }

  function _onVision(data) {
    _screenshotCount++;
    if (_screenshotsValue) _screenshotsValue.textContent = _screenshotCount;
  }

  function _onStruggle(data) {
    _struggleCount++;
    if (_strugglesValue) _strugglesValue.textContent = _struggleCount;
  }

  function _onControl(data) {
    if (!data || typeof data !== 'object') return;
    const action = data.action;
    if (action === 'session_mode') {
      const mode = data.actual_mode || data.requested_mode || '--';
      if (_sessionLabel) _sessionLabel.textContent = String(mode);

      // Update session pill styling
      const pill = document.getElementById('session-pill');
      if (pill) {
        pill.classList.remove('status-pill--muted', 'status-pill--info', 'status-pill--connected');
        if (mode === 'live') {
          pill.classList.add('status-pill--connected');
        } else if (mode === 'text') {
          pill.classList.add('status-pill--info');
        } else {
          pill.classList.add('status-pill--muted');
        }
      }
    }

    if (action === 'live_ready') {
      if (_modelLabel) _modelLabel.textContent = data.model || 'Live Flash';
      const pill = document.getElementById('model-pill');
      if (pill) {
        pill.classList.remove('status-pill--info');
        pill.classList.add('status-pill--connected');
      }
    }

    if (action === 'live_api_unavailable') {
      if (_modelLabel) _modelLabel.textContent = 'Text Only';
      if (_sessionLabel) _sessionLabel.textContent = 'text';
    }
  }

  function _onHealth(data) {
    if (!data || typeof data !== 'object') return;
    // Server-pushed health data
    const rpm = _toFiniteNumber(data.rpm, data.current_rpm, data.requests_per_minute);
    if (rpm !== null) _updateRPM(rpm);

    const model = _coerceLabel(data.model, data.live_model, data.model_name);
    if (model) _setModel(model);

    const sessionMode = _coerceLabel(data.session_mode, data.mode, data.session);
    if (sessionMode) _setSession(sessionMode);

    const pushedLatency = _toFiniteNumber(data.latency_ms, data.latency, data.ping_ms);
    if (pushedLatency !== null && _latencyLabel) {
      _latencyLabel.textContent = `${Math.round(pushedLatency)} ms`;
    }
  }

  function _onRateLimit(data) {
    if (!data || typeof data !== 'object') return;
    const rpm = _toFiniteNumber(data.current_rpm, data.rpm);
    if (rpm !== null) _updateRPM(rpm);
    _updateRateLimitStatus(_coerceLabel(data.status, data.level, data.state) || 'unknown');
  }

  // ---- RPM ----

  function _updateRPM(value) {
    const safeValue = Math.max(0, Math.round(Number(value) || 0));
    _rpm = safeValue;
    if (_rpmCurrent) _rpmCurrent.textContent = safeValue;

    const pct = Math.min(100, (safeValue / RPM_BUDGET) * 100);
    if (_rpmBar) {
      _rpmBar.style.width = `${pct}%`;
      _rpmBar.classList.remove('progress-bar__fill--warning', 'progress-bar__fill--error');
      if (pct > 83) {
        _rpmBar.classList.add('progress-bar__fill--error');
      } else if (pct > 66) {
        _rpmBar.classList.add('progress-bar__fill--warning');
      }
    }
  }

  function _updateRateLimitStatus(status) {
    if (!_ratelimitValue) return;

    const normalized = (status || 'unknown').toString().toLowerCase();

    _ratelimitValue.className = ''; // Reset
    if (normalized === 'normal') {
      _ratelimitValue.textContent = 'Normal';
      _ratelimitValue.classList.add('text-success');
    } else if (normalized === 'caution') {
      _ratelimitValue.textContent = 'Caution';
      _ratelimitValue.classList.add('text-warning');
    } else if (normalized === 'emergency') {
      _ratelimitValue.textContent = 'Emergency';
      _ratelimitValue.classList.add('text-error');
    } else if (normalized === 'critical') {
      _ratelimitValue.textContent = 'Critical';
      _ratelimitValue.classList.add('text-error');
    } else {
      _ratelimitValue.textContent = 'Unknown';
      _ratelimitValue.classList.add('text-muted');
    }
  }

  // ---- Uptime ----

  function _startUptime() {
    _stopUptime();
    _uptimeTimer = setInterval(() => {
      if (!_uptimeStart) return;
      const elapsed = Math.floor((Date.now() - _uptimeStart) / 1000);
      if (_uptimeValue) _uptimeValue.textContent = _formatDuration(elapsed);
    }, 1000);
  }

  function _stopUptime() {
    if (_uptimeTimer) {
      clearInterval(_uptimeTimer);
      _uptimeTimer = null;
    }
  }

  function _formatDuration(totalSeconds) {
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  // ---- Latency ----

  function _startLatencyPoll() {
    _stopLatencyPoll();
    _latencyTimer = setInterval(() => {
      const lat = RioSocket.getLatency();
      if (_latencyLabel) {
        if (lat !== null && lat < 30000) {
          _latencyLabel.textContent = `${lat} ms`;
        } else {
          _latencyLabel.textContent = '-- ms';
        }
      }

      // Update latency pill color
      if (lat !== null) {
        if (lat < 2000) {
          _setLatencyPillState('connected');
        } else if (lat < 5000) {
          _setLatencyPillState('connecting');
        } else {
          _setLatencyPillState('disconnected');
        }
      } else {
        _setLatencyPillState('muted');
      }
    }, 2000);
  }

  function _stopLatencyPoll() {
    if (_latencyTimer) {
      clearInterval(_latencyTimer);
      _latencyTimer = null;
    }
  }

  function _setModel(model) {
    if (_modelLabel) _modelLabel.textContent = model;
  }

  function _setSession(mode) {
    if (_sessionLabel) _sessionLabel.textContent = mode;
  }

  function _setLatencyPillState(state) {
    const pill = document.getElementById('latency-pill');
    if (!pill) return;
    pill.classList.remove('status-pill--muted', 'status-pill--connected', 'status-pill--connecting', 'status-pill--disconnected');
    if (state === 'connected') pill.classList.add('status-pill--connected');
    else if (state === 'connecting') pill.classList.add('status-pill--connecting');
    else if (state === 'disconnected') pill.classList.add('status-pill--disconnected');
    else pill.classList.add('status-pill--muted');
  }

  function _coerceLabel(...values) {
    for (const value of values) {
      if (value !== undefined && value !== null) {
        const text = String(value).trim();
        if (text) return text;
      }
    }
    return '';
  }

  function _toFiniteNumber(...values) {
    for (const value of values) {
      const n = Number(value);
      if (Number.isFinite(n)) return n;
    }
    return null;
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return {
    getMessageCount: () => _messageCount,
    getScreenshotCount: () => _screenshotCount,
    getStruggleCount: () => _struggleCount,
  };
})();
