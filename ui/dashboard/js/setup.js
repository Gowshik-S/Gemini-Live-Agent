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
// Init — load existing profiles on page load
// ═══════════════════════════════════════════════════════════════════════════
(async () => {
  const [cc, tutor] = await Promise.all([
    loadProfile('customer_care'),
    loadProfile('tutor'),
  ]);
  if (cc) populateCustomerCare(cc);
  if (tutor) populateTutor(tutor);

  // Initialize FAQ list (render empty or loaded)
  renderFaq();
})();
