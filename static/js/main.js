const SEGMENT_LABELS = {
  High_Value: "High Value",
  Loyal: "Loyal",
  Occasional: "Occasional",
  Regular: "Regular",
};

document.addEventListener("DOMContentLoaded", () => {
  initForm();
});

async function initForm() {
  const form = document.getElementById("predict-form");
  if (!form) return;

  try {
    const res = await fetch("/api/options");
    const data = await res.json();
    fillSelect("payment_method", data.payment_methods);
    fillSelect("region", data.regions);
  } catch {
    showError("Could not load options.");
  }

  form.addEventListener("submit", handleSubmit);
  document.getElementById("fill-sample")?.addEventListener("click", fillSample);
}

function fillSelect(id, options) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = options.map((o) => `<option value="${o}">${o}</option>`).join("");
}

function fillSample() {
  const sample = {
    age: 42,
    annual_income: 65000,
    months_active: 36,
    avg_monthly_spend: 450,
    purchase_frequency: 3.2,
    avg_order_value: 95,
    discount_usage_rate: 0.18,
    return_rate: 0.05,
    browsing_time_minutes: 85,
    support_interactions: 1,
    payment_method: "Card",
    region: "Urban",
  };
  Object.entries(sample).forEach(([key, val]) => {
    const el = document.querySelector(`[name="${key}"]`);
    if (el) el.value = val;
  });
}

async function handleSubmit(e) {
  e.preventDefault();
  hideError();

  const btn = document.getElementById("submit-btn");
  btn.disabled = true;
  btn.textContent = "Wait…";

  const formData = new FormData(e.target);
  const payload = Object.fromEntries(formData.entries());

  for (const key of Object.keys(payload)) {
    if (key !== "payment_method" && key !== "region") {
      payload[key] = parseFloat(payload[key]);
    }
  }
  payload.age = parseInt(payload.age, 10);
  payload.months_active = parseInt(payload.months_active, 10);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict";
  }
}

function renderResult(data) {
  const panel = document.getElementById("result-panel");
  if (!panel) return;

  const info = data.segment_info || {};
  const title = info.title || SEGMENT_LABELS[data.segment] || data.segment;

  const probs = Object.entries(data.probabilities)
    .sort(([, a], [, b]) => b - a)
    .map(([seg, pct]) => `
      <div class="prob-row">
        <span>${SEGMENT_LABELS[seg] || seg}</span>
        <strong>${pct}%</strong>
        <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
      </div>`)
    .join("");

  panel.innerHTML = `
    <div class="result-label">Prediction</div>
    <div class="result-segment">${title}</div>
    <div class="result-confidence">Confidence ${data.confidence}%</div>
    <p class="result-desc">${info.description || ""}</p>
    <div class="prob-list">${probs}</div>`;
}

function showError(msg) {
  const el = document.getElementById("error-msg");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("visible");
}

function hideError() {
  document.getElementById("error-msg")?.classList.remove("visible");
}
