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
  let _addBtn;
  let _refreshBtn;

  let _refreshInFlight = false;
  let _addInFlight = false;
  const _removeInFlight = new Set();

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

    _addBtn = document.getElementById('schedule-add');
    _refreshBtn = document.getElementById('schedule-refresh');

    _addBtn?.addEventListener('click', _add);
    _refreshBtn?.addEventListener('click', _refresh);
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
    if (_refreshInFlight) return;
    _refreshInFlight = true;
    _setButtonBusy(_refreshBtn, true);

    const user = _user();
    localStorage.setItem('rio.dashboard.user_id', user);
    _setStatus('Loading schedules...', 'text-muted');

    try {
      const resp = await fetch(`/api/triggers?user_id=${encodeURIComponent(user)}`);
      const data = await _readJson(resp);
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      const triggers = Array.isArray(data.triggers) ? data.triggers : [];
      _render(triggers, user);

      if (!data.active) {
        _setStatus('No active schedule session for this user.', 'text-warning');
      } else if (triggers.length === 0) {
        _setStatus('No schedules found for this user.', 'text-muted');
      } else {
        _setStatus(`Loaded ${triggers.length} schedule(s).`, 'text-success');
      }
    } catch (err) {
      _render([], user);
      _setStatus(`Failed to load triggers: ${String(err)}`, 'text-error');
    } finally {
      _refreshInFlight = false;
      _setButtonBusy(_refreshBtn, false);
    }
  }

  async function _add() {
    if (_addInFlight) return;

    const user = _user();
    const goal = (_goalInput?.value || '').trim();
    const intervalSeconds = Number(_intervalInput?.value || '0');
    if (!goal || !Number.isFinite(intervalSeconds) || intervalSeconds <= 0) {
      _setStatus('Goal and valid interval are required', 'text-warning');
      return;
    }

    _addInFlight = true;
    _setButtonBusy(_addBtn, true);
    _setStatus('Adding schedule...', 'text-muted');

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
      const data = await _readJson(resp);
      if (!resp.ok) {
        _setStatus(data.error || `Failed to add schedule (HTTP ${resp.status})`, 'text-error');
        return;
      }
      if (!data.success) {
        _setStatus(data.error || 'Failed to add schedule', 'text-error');
        return;
      }
      _goalInput.value = '';
      _setStatus(`Added schedule ${data.trigger?.name || ''}`, 'text-success');
      await _refresh();
    } catch (err) {
      _setStatus(`Failed to add schedule: ${String(err)}`, 'text-error');
    } finally {
      _addInFlight = false;
      _setButtonBusy(_addBtn, false);
    }
  }

  async function _remove(name, buttonEl) {
    if (!name || _removeInFlight.has(name)) return;

    _removeInFlight.add(name);
    _setButtonBusy(buttonEl, true);
    _setStatus(`Removing ${name}...`, 'text-muted');

    const user = _user();
    try {
      const resp = await fetch(`/api/triggers/${encodeURIComponent(name)}?user_id=${encodeURIComponent(user)}`, {
        method: 'DELETE',
      });
      const data = await _readJson(resp);
      if (!resp.ok) {
        _setStatus(data.error || `Failed to remove schedule (HTTP ${resp.status})`, 'text-error');
        return;
      }
      if (!data.success) {
        _setStatus(data.error || 'Failed to remove schedule', 'text-error');
        return;
      }
      _setStatus(`Removed ${name}`, 'text-success');
      await _refresh();
    } catch (err) {
      _setStatus(`Failed to remove schedule: ${String(err)}`, 'text-error');
    } finally {
      _removeInFlight.delete(name);
      _setButtonBusy(buttonEl, false);
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
      del.addEventListener('click', () => _remove(item.name, del));

      row.appendChild(meta);
      row.appendChild(del);
      _listEl.appendChild(row);
    });
  }

  function _setButtonBusy(buttonEl, busy) {
    if (!buttonEl) return;
    buttonEl.disabled = !!busy;
  }

  async function _readJson(resp) {
    try {
      return await resp.json();
    } catch {
      return {};
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { refresh: _refresh };
})();
