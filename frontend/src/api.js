// api.js
// -------
// One tiny module that knows how to talk to the FastAPI backend.
//
// Base URL resolution:
//   - In local dev, VITE_API_URL is unset, so we use "/api" and let Vite's dev
//     proxy (see vite.config.js) forward to http://localhost:8000.
//   - In production (Vercel/Render), set VITE_API_URL to the deployed backend
//     URL, e.g. https://churn-api.onrender.com
//
// Render's blueprint injects just the hostname (no scheme), so we normalise:
// if VITE_API_URL is set without a leading http(s), we prepend https://.
function resolveBase() {
  const env = import.meta.env.VITE_API_URL;
  if (!env) return "/api"; // local dev: use the Vite proxy
  return /^https?:\/\//.test(env) ? env : `https://${env}`;
}
const BASE = resolveBase();

async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed (${res.status})`);
  return res.json();
}

// GET /metrics -> { summary, model_comparison }
export const fetchMetrics = () => getJSON("/metrics");

// POST /predict -> scoring result for one customer
export async function predictCustomer(customer) {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Prediction failed (${res.status})`);
  }
  return res.json();
}

// POST /insight -> ML prediction + a Gemini (or fallback) plain-English insight.
// The probability still comes from the ML model; Gemini only explains it in words.
export async function fetchInsight(customer) {
  const res = await fetch(`${BASE}/insight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Insight failed (${res.status})`);
  }
  return res.json();
}

// POST /predict-batch -> returns a CSV blob (the scored file) to download
export async function predictBatch(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/predict-batch`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Batch scoring failed (${res.status})`);
  }
  return res.blob();
}
