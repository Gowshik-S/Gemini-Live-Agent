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
  }

  function _onDisconnect() {
    _stopUptime();
    _stopLatencyPoll();
    if (_latencyLabel) _latencyLabel.textContent = '-- ms';
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
    const action = data.action;
    if (action === 'session_mode') {
      const mode = data.actual_mode || data.requested_mode || '--';
      if (_sessionLabel) _sessionLabel.textContent = mode;

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
      if (_modelLabel) _modelLabel.textContent = 'Live Flash';
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
    // Server-pushed health data
    if (data.rpm !== undefined) _updateRPM(data.rpm);
    if (data.model) {
      if (_modelLabel) _modelLabel.textContent = data.model;
    }
  }

  function _onRateLimit(data) {
    if (data.current_rpm !== undefined) _updateRPM(data.current_rpm);
    if (data.status) _updateRateLimitStatus(data.status);
  }

  // ---- RPM ----

  function _updateRPM(value) {
    _rpm = value;
    if (_rpmCurrent) _rpmCurrent.textContent = value;

    const pct = Math.min(100, (value / RPM_BUDGET) * 100);
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

    _ratelimitValue.className = ''; // Reset
    if (status === 'normal') {
      _ratelimitValue.textContent = 'Normal';
      _ratelimitValue.classList.add('text-success');
    } else if (status === 'caution') {
      _ratelimitValue.textContent = 'Caution';
      _ratelimitValue.classList.add('text-warning');
    } else if (status === 'emergency') {
      _ratelimitValue.textContent = 'Emergency';
      _ratelimitValue.classList.add('text-error');
    } else if (status === 'critical') {
      _ratelimitValue.textContent = 'Critical';
      _ratelimitValue.classList.add('text-error');
    } else {
      _ratelimitValue.textContent = status;
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
      const pill = document.getElementById('latency-pill');
      if (pill && lat !== null) {
        pill.classList.remove('status-pill--muted', 'status-pill--connected', 'status-pill--connecting', 'status-pill--disconnected');
        if (lat < 2000) {
          pill.classList.add('status-pill--connected');
        } else if (lat < 5000) {
          pill.classList.add('status-pill--connecting');
        } else {
          pill.classList.add('status-pill--disconnected');
        }
      }
    }, 2000);
  }

  function _stopLatencyPoll() {
    if (_latencyTimer) {
      clearInterval(_latencyTimer);
      _latencyTimer = null;
    }
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
