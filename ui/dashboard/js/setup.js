/**
 * Rio Setup Page — Form logic
 *
 * Fixed schema approach: the fields are predetermined.
 * Users fill them in via the form, we save as JSON via the API.
 * NO LLM generation of config — everything is deterministic.
 */

// ═══════════════════════════════════════════════════════════════════════════
// API base — auto-detect from current location
// ═══════════════════════════════════════════════════════════════════════════
const API_BASE = window.location.origin;
const DEFAULT_SETUP_USER_ID = 'default';
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

let translatorUserConfigCache = null;

// ═══════════════════════════════════════════════════════════════════════════
// Tabs
// ═══════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-btn--active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('tab-content--active'));
    btn.classList.add('tab-btn--active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('tab-content--active');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Toast
// ═══════════════════════════════════════════════════════════════════════════
function showToast(elementId, message, isError = false) {
  const toast = document.getElementById(elementId);
  toast.querySelector('.toast__msg').textContent = message;
  toast.querySelector('.toast__icon').textContent = isError ? '✗' : '✓';
  toast.className = isError ? 'toast toast--error' : 'toast';
  toast.style.display = 'flex';
  setTimeout(() => { toast.style.display = 'none'; }, 3000);
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: textarea ↔ array (one item per line)
// ═══════════════════════════════════════════════════════════════════════════
function linesToArray(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  return el.value.split('\n').map(s => s.trim()).filter(Boolean);
}

function arrayToLines(id, arr) {
  const el = document.getElementById(id);
  if (!el || !arr) return;
  el.value = (arr || []).join('\n');
}

// ═══════════════════════════════════════════════════════════════════════════
// Toggle-driven config visibility
// ═══════════════════════════════════════════════════════════════════════════
function setProfileConfigVisibility(formId, enabled) {
  const form = document.getElementById(formId);
  if (!form) return;

  form.querySelectorAll('.form-section').forEach(section => {
    section.style.display = enabled ? '' : 'none';
  });
}

function syncCustomerCareVisibility() {
  const enabled = document.getElementById('cc-enabled').checked;
  setProfileConfigVisibility('form-customer-care', enabled);
}

function syncTutorVisibility() {
  const enabled = document.getElementById('tutor-enabled').checked;
  setProfileConfigVisibility('form-tutor', enabled);
}

// ═══════════════════════════════════════════════════════════════════════════
// FAQ dynamic list
// ═══════════════════════════════════════════════════════════════════════════
let faqItems = [];

function renderFaq() {
  const list = document.getElementById('faq-list');
  list.innerHTML = '';
  faqItems.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'faq-item';
    div.innerHTML = `
      <div class="faq-item__fields">
        <input type="text" placeholder="Question" value="${escapeHtml(item.q)}" data-faq="${i}" data-field="q">
        <input type="text" placeholder="Answer" value="${escapeHtml(item.a)}" data-faq="${i}" data-field="a">
      </div>
      <button type="button" class="btn-remove" data-faq-remove="${i}">×</button>
    `;
    list.appendChild(div);
  });

  // Bind events
  list.querySelectorAll('input[data-faq]').forEach(inp => {
    inp.addEventListener('input', e => {
      const idx = parseInt(e.target.dataset.faq);
      const field = e.target.dataset.field;
      faqItems[idx][field] = e.target.value;
    });
  });
  list.querySelectorAll('[data-faq-remove]').forEach(btn => {
    btn.addEventListener('click', e => {
      faqItems.splice(parseInt(e.target.dataset.faqRemove), 1);
      renderFaq();
    });
  });
}

document.getElementById('btn-add-faq').addEventListener('click', () => {
  faqItems.push({ q: '', a: '' });
  renderFaq();
});

function escapeHtml(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ═══════════════════════════════════════════════════════════════════════════
// Customer Care: form → JSON
// ═══════════════════════════════════════════════════════════════════════════
function collectCustomerCare() {
  return {
    version: '1.0',
    enabled: document.getElementById('cc-enabled').checked,
    business: {
      business_name: document.getElementById('cc-business-name').value.trim(),
      tagline: document.getElementById('cc-tagline').value.trim(),
      industry: document.getElementById('cc-industry').value,
      website: document.getElementById('cc-website').value.trim(),
      products_services: linesToArray('cc-products'),
      return_policy: document.getElementById('cc-return-policy').value.trim(),
      refund_policy: document.getElementById('cc-refund-policy').value.trim(),
      shipping_policy: document.getElementById('cc-shipping-policy').value.trim(),
      warranty_policy: document.getElementById('cc-warranty-policy').value.trim(),
      privacy_note: '',
      business_hours: document.getElementById('cc-hours').value.trim(),
      support_channels: linesToArray('cc-channels'),
      escalation_email: document.getElementById('cc-escalation-email').value.trim(),
      escalation_phone: document.getElementById('cc-escalation-phone').value.trim(),
      sla_response_time: document.getElementById('cc-sla').value.trim(),
      faq: faqItems.filter(f => f.q || f.a),
      tone: document.getElementById('cc-tone').value,
      language: document.getElementById('cc-language').value.trim(),
      greeting: document.getElementById('cc-greeting').value.trim(),
      out_of_scope_topics: linesToArray('cc-out-of-scope'),
      redirect_message: document.getElementById('cc-redirect').value.trim(),
    },
    tier_rules: {
      tier_0: document.getElementById('cc-tier-0').value.trim(),
      tier_1: document.getElementById('cc-tier-1').value.trim(),
      tier_2: document.getElementById('cc-tier-2').value.trim(),
      tier_3: document.getElementById('cc-tier-3').value.trim(),
    },
  };
}

function populateCustomerCare(data) {
  if (!data) return;
  const b = data.business || {};

  document.getElementById('cc-enabled').checked = data.enabled !== false;
  document.getElementById('cc-business-name').value = b.business_name || '';
  document.getElementById('cc-tagline').value = b.tagline || '';
  document.getElementById('cc-industry').value = b.industry || '';
  document.getElementById('cc-website').value = b.website || '';
  arrayToLines('cc-products', b.products_services);
  document.getElementById('cc-return-policy').value = b.return_policy || '';
  document.getElementById('cc-refund-policy').value = b.refund_policy || '';
  document.getElementById('cc-shipping-policy').value = b.shipping_policy || '';
  document.getElementById('cc-warranty-policy').value = b.warranty_policy || '';
  document.getElementById('cc-hours').value = b.business_hours || '';
  arrayToLines('cc-channels', b.support_channels);
  document.getElementById('cc-escalation-email').value = b.escalation_email || '';
  document.getElementById('cc-escalation-phone').value = b.escalation_phone || '';
  document.getElementById('cc-sla').value = b.sla_response_time || '';
  document.getElementById('cc-tone').value = b.tone || 'professional and friendly';
  document.getElementById('cc-language').value = b.language || 'English';
  document.getElementById('cc-greeting').value = b.greeting || '';
  arrayToLines('cc-out-of-scope', b.out_of_scope_topics);
  document.getElementById('cc-redirect').value = b.redirect_message || '';

  // Tier rules
  const t = data.tier_rules || {};
  if (t.tier_0) document.getElementById('cc-tier-0').value = t.tier_0;
  if (t.tier_1) document.getElementById('cc-tier-1').value = t.tier_1;
  if (t.tier_2) document.getElementById('cc-tier-2').value = t.tier_2;
  if (t.tier_3) document.getElementById('cc-tier-3').value = t.tier_3;

  // FAQ
  faqItems = (b.faq || []).map(f => ({ q: f.q || '', a: f.a || '' }));
  renderFaq();
  syncCustomerCareVisibility();
}

// ═══════════════════════════════════════════════════════════════════════════
// Tutor: form → JSON
// ═══════════════════════════════════════════════════════════════════════════
function collectTutor() {
  const modeEl = document.querySelector('input[name="homework-mode"]:checked');
  return {
    version: '1.0',
    enabled: document.getElementById('tutor-enabled').checked,
    student: {
      student_name: document.getElementById('tutor-name').value.trim(),
      grade_level: document.getElementById('tutor-grade').value.trim(),
      age_group: document.getElementById('tutor-age').value,
      subjects: linesToArray('tutor-subjects'),
      current_courses: linesToArray('tutor-courses'),
      learning_goals: linesToArray('tutor-goals'),
      strengths: linesToArray('tutor-strengths'),
      weaknesses: linesToArray('tutor-weaknesses'),
      learning_style: document.getElementById('tutor-style').value,
      preferred_difficulty: document.getElementById('tutor-difficulty').value,
      language: document.getElementById('tutor-language').value.trim(),
      topics_off_limits: linesToArray('tutor-off-limits'),
      homework_help_mode: modeEl ? modeEl.value : 'guide',
    },
    socratic_mode: document.getElementById('tutor-socratic').checked,
  };
}

function populateTutor(data) {
  if (!data) return;
  const s = data.student || {};

  document.getElementById('tutor-enabled').checked = data.enabled !== false;
  document.getElementById('tutor-name').value = s.student_name || '';
  document.getElementById('tutor-grade').value = s.grade_level || '';
  document.getElementById('tutor-age').value = s.age_group || '';
  arrayToLines('tutor-subjects', s.subjects);
  arrayToLines('tutor-courses', s.current_courses);
  arrayToLines('tutor-goals', s.learning_goals);
  arrayToLines('tutor-strengths', s.strengths);
  arrayToLines('tutor-weaknesses', s.weaknesses);
  document.getElementById('tutor-style').value = s.learning_style || 'visual';
  document.getElementById('tutor-difficulty').value = s.preferred_difficulty || 'intermediate';
  document.getElementById('tutor-language').value = s.language || 'English';
  arrayToLines('tutor-off-limits', s.topics_off_limits);
  document.getElementById('tutor-socratic').checked = data.socratic_mode !== false;

  // Homework mode radios
  const mode = s.homework_help_mode || 'guide';
  const radio = document.querySelector(`input[name="homework-mode"][value="${mode}"]`);
  if (radio) radio.checked = true;
  syncTutorVisibility();
}

// ═══════════════════════════════════════════════════════════════════════════
// API calls
// ═══════════════════════════════════════════════════════════════════════════
async function loadProfile(skillName) {
  try {
    const res = await fetch(`${API_BASE}/api/profiles/${skillName}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.profile || null;
  } catch (err) {
    console.warn(`Failed to load ${skillName} profile:`, err);
    return null;
  }
}

async function saveProfile(skillName, data) {
  const res = await fetch(`${API_BASE}/api/profiles/${skillName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return await res.json();
}

// ═══════════════════════════════════════════════════════════════════════════
// Form submissions
// ═══════════════════════════════════════════════════════════════════════════
document.getElementById('form-customer-care').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const data = collectCustomerCare();
    await saveProfile('customer_care', data);
    showToast('toast-cc', 'Customer care profile saved!');
  } catch (err) {
    showToast('toast-cc', `Error: ${err.message}`, true);
  }
});

document.getElementById('form-tutor').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const data = collectTutor();
    await saveProfile('tutor', data);
    showToast('toast-tutor', 'Tutor profile saved!');
  } catch (err) {
    showToast('toast-tutor', `Error: ${err.message}`, true);
  }
});

// Reset buttons
document.getElementById('btn-reset-cc').addEventListener('click', async () => {
  try {
    const res = await fetch(`${API_BASE}/api/profiles/customer_care/defaults`);
    const data = await res.json();
    populateCustomerCare(data.profile);
    showToast('toast-cc', 'Reset to defaults');
  } catch (err) {
    showToast('toast-cc', 'Failed to load defaults', true);
  }
});

document.getElementById('btn-reset-tutor').addEventListener('click', async () => {
  try {
    const res = await fetch(`${API_BASE}/api/profiles/tutor/defaults`);
    const data = await res.json();
    populateTutor(data.profile);
    showToast('toast-tutor', 'Reset to defaults');
  } catch (err) {
    showToast('toast-tutor', 'Failed to load defaults', true);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// Agent Settings
// ═══════════════════════════════════════════════════════════════════════════

async function loadAgentSettings() {
  try {
    const res = await fetch(`${API_BASE}/api/settings`);
    if (!res.ok) return;
    const data = await res.json();
    const s = data.settings || {};
    document.getElementById('settings-name').value = s.agent_name || 'Rio';
    document.getElementById('settings-tagline').value = s.agent_tagline || '';
    document.getElementById('settings-role').value = s.agent_role || 'assistant';
  } catch (err) {
    console.warn('Failed to load agent settings:', err);
  }
}

async function loadModelsStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/models/status`);
    if (!res.ok) return;
    const data = await res.json();

    // API Key badge
    const apiEl = document.getElementById('settings-api-status');
    if (apiEl) {
      apiEl.textContent = data.api_key_set ? 'Configured' : 'Missing';
      apiEl.className = 'status-badge ' + (data.api_key_set ? 'status-badge--ok' : 'status-badge--missing');
    }

    // Vertex AI badge
    const vxEl = document.getElementById('settings-vertex-status');
    if (vxEl) {
      vxEl.textContent = data.vertex_ai ? 'Enabled' : 'Disabled';
      vxEl.className = 'status-badge ' + (data.vertex_ai ? 'status-badge--ok' : 'status-badge--muted');
    }

    // Model badges
    const models = data.models || {};
    const liveEl = document.getElementById('settings-live-model');
    if (liveEl) liveEl.textContent = models.live || '--';
    const orchEl = document.getElementById('settings-orch-model');
    if (orchEl) orchEl.textContent = models.orchestrator || '--';
  } catch (err) {
    console.warn('Failed to load model status:', err);
  }
}

document.getElementById('form-agent-settings').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const body = {
      agent_name: document.getElementById('settings-name').value.trim(),
      agent_tagline: document.getElementById('settings-tagline').value.trim(),
      agent_role: document.getElementById('settings-role').value,
    };
    const res = await fetch(`${API_BASE}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    showToast('toast-settings', 'Settings saved! Restart the server for changes to take effect.');
  } catch (err) {
    showToast('toast-settings', `Error: ${err.message}`, true);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// Agent Orchestration Config
// ═══════════════════════════════════════════════════════════════════════════
const AGENT_TYPES = ['task', 'code', 'research', 'creative'];
const AGENT_KEY_MAP = {
  task: 'task_executor',
  code: 'code_agent',
  research: 'research_agent',
  creative: 'creative_agent',
};

function collectAgents() {
  const agents = {};
  for (const type of AGENT_TYPES) {
    const key = AGENT_KEY_MAP[type];
    agents[key] = {
      enabled: document.getElementById(`agent-${type}-enabled`).checked,
      model: document.getElementById(`agent-${type}-model`).value,
      tools: document.getElementById(`agent-${type}-tools`).value,
      max_iterations: parseInt(document.getElementById(`agent-${type}-iterations`).value) || 10,
    };
  }
  return agents;
}

function populateAgents(data) {
  if (!data) return;
  for (const type of AGENT_TYPES) {
    const key = AGENT_KEY_MAP[type];
    const agent = data[key];
    if (!agent) continue;
    const enabledEl = document.getElementById(`agent-${type}-enabled`);
    const modelEl = document.getElementById(`agent-${type}-model`);
    const toolsEl = document.getElementById(`agent-${type}-tools`);
    const iterEl = document.getElementById(`agent-${type}-iterations`);
    if (enabledEl) enabledEl.checked = agent.enabled !== false;
    if (modelEl) modelEl.value = agent.model || 'gemini-2.5-flash';
    if (toolsEl) toolsEl.value = agent.tools || 'all';
    if (iterEl) iterEl.value = agent.max_iterations || 10;
  }
}

async function loadAgents() {
  try {
    const res = await fetch(`${API_BASE}/api/agents`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.agents || null;
  } catch (err) {
    console.warn('Failed to load agents config:', err);
    return null;
  }
}

document.getElementById('form-agents').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const agents = collectAgents();
    const res = await fetch(`${API_BASE}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agents }),
    });
    if (!res.ok) throw new Error(await res.text());
    showToast('toast-agents', 'Agent configuration saved!');
  } catch (err) {
    showToast('toast-agents', `Error: ${err.message}`, true);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// Translator Settings
// ═══════════════════════════════════════════════════════════════════════════

function getTranslatorUserId() {
  const el = document.getElementById('tr-user-id');
  if (!el) return DEFAULT_SETUP_USER_ID;
  const raw = (el.value || '').trim().toLowerCase();
  return raw || DEFAULT_SETUP_USER_ID;
}

function setTranslatorUserId(userId) {
  const el = document.getElementById('tr-user-id');
  if (!el) return;
  el.value = userId || DEFAULT_SETUP_USER_ID;
}

function populateLanguageSelect(selectId, defaultCode = 'en', includeAuto = false) {
  const el = document.getElementById(selectId);
  if (!el) return;
  el.innerHTML = '';

  const options = includeAuto ? LANGUAGE_OPTIONS : LANGUAGE_OPTIONS.filter((item) => item.code !== 'auto');
  options.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.code;
    opt.textContent = `${item.label} (${item.code})`;
    if (item.code === defaultCode) opt.selected = true;
    el.appendChild(opt);
  });
}

function normalizeTranslatorConfig(raw) {
  const cfg = (raw && typeof raw === 'object') ? raw : {};
  return {
    enabled: Boolean(cfg.enabled),
    source_language: String(cfg.source_language || cfg.counterparty_language || 'en').toLowerCase(),
    target_language: String(cfg.target_language || cfg.staff_language || 'en').toLowerCase(),
    staff_language: String(cfg.staff_language || cfg.target_language || 'en').toLowerCase(),
    counterparty_language: String(cfg.counterparty_language || cfg.source_language || 'en').toLowerCase(),
    bidirectional: cfg.bidirectional !== false,
    speak_translation: cfg.speak_translation !== false,
    suppress_task_detection: cfg.suppress_task_detection !== false,
    model: cfg.model || 'gemini-2.5-flash',
    repeat_guard_ms: Number(cfg.repeat_guard_ms || 1200),
    min_chars: Number(cfg.min_chars || 2),
    timeout_seconds: Number(cfg.timeout_seconds || 6.0),
  };
}

function populateTranslatorForm(config) {
  const tr = normalizeTranslatorConfig(config);
  document.getElementById('tr-enabled').checked = Boolean(tr.enabled);
  document.getElementById('tr-staff-language').value = tr.staff_language || 'en';
  document.getElementById('tr-counterparty-language').value = tr.counterparty_language || 'en';
  document.getElementById('tr-bidirectional').checked = Boolean(tr.bidirectional);
  document.getElementById('tr-speak').checked = Boolean(tr.speak_translation);
  document.getElementById('tr-suppress-tasks').checked = Boolean(tr.suppress_task_detection);
}

function collectTranslatorForm() {
  const staffLanguage = (document.getElementById('tr-staff-language').value || 'en').toLowerCase();
  const counterpartyLanguage = (document.getElementById('tr-counterparty-language').value || 'en').toLowerCase();
  return {
    enabled: document.getElementById('tr-enabled').checked,
    source_language: counterpartyLanguage,
    target_language: staffLanguage,
    staff_language: staffLanguage,
    counterparty_language: counterpartyLanguage,
    bidirectional: document.getElementById('tr-bidirectional').checked,
    speak_translation: document.getElementById('tr-speak').checked,
    suppress_task_detection: document.getElementById('tr-suppress-tasks').checked,
  };
}

async function loadTranslatorSettings(userId) {
  try {
    const safeUser = encodeURIComponent(userId || DEFAULT_SETUP_USER_ID);
    const res = await fetch(`${API_BASE}/api/users/${safeUser}/config`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    translatorUserConfigCache = (data && data.config && typeof data.config === 'object') ? data.config : {};
    populateTranslatorForm(translatorUserConfigCache.live_translator || {});
    return translatorUserConfigCache;
  } catch (err) {
    showToast('toast-translator', `Error: ${err.message}`, true);
    return null;
  }
}

async function fetchTranslatorUserConfig(userId) {
  const safeUser = encodeURIComponent(userId || DEFAULT_SETUP_USER_ID);
  const res = await fetch(`${API_BASE}/api/users/${safeUser}/config`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return (data && data.config && typeof data.config === 'object') ? data.config : {};
}

async function saveTranslatorSettings(userId) {
  const safeUser = encodeURIComponent(userId || DEFAULT_SETUP_USER_ID);
  const latest = translatorUserConfigCache || (await fetchTranslatorUserConfig(userId));
  const mergedTranslator = {
    ...normalizeTranslatorConfig(latest.live_translator || {}),
    ...collectTranslatorForm(),
  };
  const mergedConfig = {
    ...latest,
    live_translator: mergedTranslator,
  };

  const res = await fetch(`${API_BASE}/api/users/${safeUser}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(mergedConfig),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const data = await res.json();
  translatorUserConfigCache = (data && data.config && typeof data.config === 'object') ? data.config : mergedConfig;
  populateTranslatorForm(translatorUserConfigCache.live_translator || {});
  return translatorUserConfigCache;
}

document.getElementById('form-translator').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await saveTranslatorSettings(getTranslatorUserId());
    showToast('toast-translator', 'Translator settings saved!');
  } catch (err) {
    showToast('toast-translator', `Error: ${err.message}`, true);
  }
});

document.getElementById('btn-reload-translator').addEventListener('click', async () => {
  await loadTranslatorSettings(getTranslatorUserId());
  showToast('toast-translator', 'Translator settings reloaded');
});

document.getElementById('tr-user-id').addEventListener('change', async () => {
  const userId = getTranslatorUserId();
  setTranslatorUserId(userId);
  await loadTranslatorSettings(userId);
});

// ═══════════════════════════════════════════════════════════════════════════
// Init — load existing profiles on page load
// ═══════════════════════════════════════════════════════════════════════════
(async () => {
  // Keep config sections hidden until the corresponding skill is enabled.
  document.getElementById('cc-enabled').addEventListener('change', syncCustomerCareVisibility);
  document.getElementById('tutor-enabled').addEventListener('change', syncTutorVisibility);

  populateLanguageSelect('tr-staff-language', 'en', false);
  populateLanguageSelect('tr-counterparty-language', 'en', true);

  const params = new URLSearchParams(window.location.search);
  const initialUser = (params.get('user_id') || params.get('user') || DEFAULT_SETUP_USER_ID).trim().toLowerCase() || DEFAULT_SETUP_USER_ID;
  setTranslatorUserId(initialUser);

  const [cc, tutor, , , agents] = await Promise.all([
    loadProfile('customer_care'),
    loadProfile('tutor'),
    loadAgentSettings(),
    loadModelsStatus(),
    loadAgents(),
  ]);
  await loadTranslatorSettings(initialUser);
  if (cc) populateCustomerCare(cc);
  if (tutor) populateTutor(tutor);
  if (agents) populateAgents(agents);

  // Apply initial visibility when no profile is loaded or defaults are active.
  syncCustomerCareVisibility();
  syncTutorVisibility();

  // Initialize FAQ list (render empty or loaded)
  renderFaq();
})();
