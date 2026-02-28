/* ==========================================================================
   Rio Dashboard — Struggle Confidence Gauge
   SVG arc gauge with needle + signal indicators.
   ========================================================================== */

'use strict';

const RioGauge = (() => {
  // Arc geometry (matches the SVG path in index.html)
  const ARC_LENGTH = 251.2; // Approximate length of the semicircular arc

  let _arc = null;
  let _needle = null;
  let _valueEl = null;
  let _labelEl = null;
  let _signalEls = {};

  let _currentValue = 0;
  let _targetValue = 0;
  let _animFrame = null;

  // Thresholds for color coding
  const LEVEL_LOW = 0.4;
  const LEVEL_MED = 0.7;
  const LEVEL_HIGH = 0.85;

  function init() {
    _arc = document.getElementById('gauge-arc');
    _needle = document.getElementById('gauge-needle');
    _valueEl = document.getElementById('gauge-value');
    _labelEl = document.getElementById('gauge-label');

    // Cache signal elements
    document.querySelectorAll('.signal-item').forEach(el => {
      const key = el.dataset.signal;
      if (key) _signalEls[key] = el;
    });

    // Listen for struggle events
    RioSocket.on('struggle', (data) => {
      const confidence = data.confidence || 0;
      const signals = data.signals || [];
      update(confidence, signals);
    });
  }

  /**
   * Update the gauge to a new confidence value and highlight active signals.
   * @param {number} confidence - 0.0 to 1.0
   * @param {string[]} signals - Active signal names
   */
  function update(confidence, signals) {
    confidence = Math.max(0, Math.min(1, confidence));
    _targetValue = confidence;

    // Animate to target
    if (!_animFrame) _animate();

    // Update label
    if (_labelEl) {
      if (confidence < LEVEL_LOW) {
        _labelEl.textContent = 'No signals';
      } else if (confidence < LEVEL_MED) {
        _labelEl.textContent = 'Low activity';
      } else if (confidence < LEVEL_HIGH) {
        _labelEl.textContent = 'Monitoring';
      } else {
        _labelEl.textContent = 'Struggling detected';
      }
    }

    // Update signal indicators
    const activeSet = new Set(signals);
    for (const [key, el] of Object.entries(_signalEls)) {
      el.classList.remove('signal-item--active', 'signal-item--high');
      if (activeSet.has(key)) {
        el.classList.add(confidence >= LEVEL_HIGH ? 'signal-item--high' : 'signal-item--active');
      }
    }
  }

  function _animate() {
    const SPEED = 0.08; // Easing speed
    const diff = _targetValue - _currentValue;

    if (Math.abs(diff) < 0.002) {
      _currentValue = _targetValue;
      _render(_currentValue);
      _animFrame = null;
      return;
    }

    _currentValue += diff * SPEED;
    _render(_currentValue);
    _animFrame = requestAnimationFrame(_animate);
  }

  function _render(value) {
    // Arc: offset from full (251.2 = hidden, 0 = full)
    if (_arc) {
      const offset = ARC_LENGTH * (1 - value);
      _arc.setAttribute('stroke-dashoffset', offset.toFixed(1));
    }

    // Needle: rotate from -90deg (0.0) to +90deg (1.0)
    if (_needle) {
      const angle = -90 + (value * 180);
      _needle.setAttribute('transform', `rotate(${angle.toFixed(1)}, 100, 100)`);
    }

    // Value text
    if (_valueEl) {
      _valueEl.textContent = value.toFixed(2);

      // Color code
      if (value < LEVEL_LOW) {
        _valueEl.style.color = 'var(--text-primary)';
      } else if (value < LEVEL_MED) {
        _valueEl.style.color = 'var(--success)';
      } else if (value < LEVEL_HIGH) {
        _valueEl.style.color = 'var(--warning)';
      } else {
        _valueEl.style.color = 'var(--error)';
      }
    }
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { update };
})();
