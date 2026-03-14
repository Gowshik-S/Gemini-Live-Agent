/* ==========================================================================
   Rio Dashboard — Schedule Manager (Priority 4.3 UI)
   ========================================================================== */

'use strict';

const RioSchedules = (() => {
  let _userInput;
  let _goalInput;
  let _intervalInput;
  let _statusEl;
  let _listEl;
  let _countEl;

  function init() {
    _userInput = document.getElementById('schedule-user-id');
    _goalInput = document.getElementById('schedule-goal');
    _intervalInput = document.getElementById('schedule-interval');
    _statusEl = document.getElementById('schedule-status');
    _listEl = document.getElementById('schedule-list');
    _countEl = document.getElementById('schedule-count');

    if (!_userInput || !_goalInput || !_intervalInput || !_statusEl || !_listEl || !_countEl) {
      return;
    }

    const savedUser = localStorage.getItem('rio.dashboard.user_id');
    if (savedUser) _userInput.value = savedUser;

    document.getElementById('schedule-add')?.addEventListener('click', _add);
    document.getElementById('schedule-refresh')?.addEventListener('click', _refresh);
    _userInput.addEventListener('change', () => {
      localStorage.setItem('rio.dashboard.user_id', _user());
      _refresh();
    });

    _refresh();
  }

  function _user() {
    const value = (_userInput?.value || 'default').trim();
    return value || 'default';
  }

  function _setStatus(text, cls = 'text-muted') {
    if (!_statusEl) return;
    _statusEl.className = `schedule-status ${cls}`;
    _statusEl.textContent = text;
  }

  async function _refresh() {
    const user = _user();
    localStorage.setItem('rio.dashboard.user_id', user);
    try {
      const resp = await fetch(`/api/triggers?user_id=${encodeURIComponent(user)}`);
      const data = await resp.json();
      const triggers = Array.isArray(data.triggers) ? data.triggers : [];
      _render(triggers, user);
      _setStatus(data.active ? `Loaded ${triggers.length} trigger(s)` : 'No active session for this user', data.active ? 'text-success' : 'text-warning');
    } catch (err) {
      _render([], user);
      _setStatus(`Failed to load triggers: ${String(err)}`, 'text-error');
    }
  }

  async function _add() {
    const user = _user();
    const goal = (_goalInput?.value || '').trim();
    const intervalSeconds = Number(_intervalInput?.value || '0');
    if (!goal || !Number.isFinite(intervalSeconds) || intervalSeconds <= 0) {
      _setStatus('Goal and valid interval are required', 'text-warning');
      return;
    }
    try {
      const resp = await fetch('/api/triggers/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user,
          goal,
          interval_seconds: intervalSeconds,
        }),
      });
      const data = await resp.json();
      if (!data.success) {
        _setStatus(data.error || 'Failed to add schedule', 'text-error');
        return;
      }
      _goalInput.value = '';
      _setStatus(`Added schedule ${data.trigger?.name || ''}`, 'text-success');
      await _refresh();
    } catch (err) {
      _setStatus(`Failed to add schedule: ${String(err)}`, 'text-error');
    }
  }

  async function _remove(name) {
    const user = _user();
    try {
      const resp = await fetch(`/api/triggers/${encodeURIComponent(name)}?user_id=${encodeURIComponent(user)}`, {
        method: 'DELETE',
      });
      const data = await resp.json();
      if (!data.success) {
        _setStatus(data.error || 'Failed to remove schedule', 'text-error');
        return;
      }
      _setStatus(`Removed ${name}`, 'text-success');
      await _refresh();
    } catch (err) {
      _setStatus(`Failed to remove schedule: ${String(err)}`, 'text-error');
    }
  }

  function _render(triggers, user) {
    if (!_listEl || !_countEl) return;
    _listEl.innerHTML = '';
    _countEl.textContent = String(triggers.length);

    if (!triggers.length) {
      const empty = document.createElement('div');
      empty.className = 'text-muted';
      empty.style.fontSize = '12px';
      empty.textContent = `No schedules for user: ${user}`;
      _listEl.appendChild(empty);
      return;
    }

    triggers.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'schedule-item';

      const meta = document.createElement('div');
      meta.className = 'schedule-item__meta';

      const name = document.createElement('div');
      name.className = 'schedule-item__name';
      name.textContent = `${item.name || 'unnamed'} • ${item.type || 'schedule'}`;

      const goal = document.createElement('div');
      goal.className = 'schedule-item__goal';
      goal.textContent = item.goal || '';

      meta.appendChild(name);
      meta.appendChild(goal);

      const del = document.createElement('button');
      del.className = 'schedule-item__delete';
      del.textContent = 'Delete';
      del.addEventListener('click', () => _remove(item.name));

      row.appendChild(meta);
      row.appendChild(del);
      _listEl.appendChild(row);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { refresh: _refresh };
})();
