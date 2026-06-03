import { useState, useEffect, useRef } from "react";
import Card from "../components/Card.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import { predictCustomer, fetchInsight } from "../api.js";

// The 14 fields the brief asks the form to collect. Each entry describes how to
// render the input: a select with options, or a number box.
const FIELDS = [
  { name: "tenure", label: "Tenure (months)", type: "number", default: 5 },
  { name: "MonthlyCharges", label: "Monthly charges", type: "number", step: "0.05", default: 85 },
  { name: "TotalCharges", label: "Total charges", type: "number", step: "0.05", default: 425 },
  { name: "Contract", label: "Contract", options: ["Month-to-month", "One year", "Two year"], default: "Month-to-month" },
  { name: "InternetService", label: "Internet service", options: ["DSL", "Fiber optic", "No"], default: "Fiber optic" },
  { name: "TechSupport", label: "Tech support", options: ["Yes", "No", "No internet service"], default: "No" },
  { name: "OnlineSecurity", label: "Online security", options: ["Yes", "No", "No internet service"], default: "No" },
  { name: "PaymentMethod", label: "Payment method", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], default: "Electronic check" },
  { name: "PaperlessBilling", label: "Paperless billing", options: ["Yes", "No"], default: "Yes" },
  { name: "SeniorCitizen", label: "Senior citizen", options: [{ v: 0, t: "No" }, { v: 1, t: "Yes" }], default: 0 },
  { name: "Partner", label: "Partner", options: ["Yes", "No"], default: "No" },
  { name: "Dependents", label: "Dependents", options: ["Yes", "No"], default: "No" },
  { name: "gender", label: "Gender", options: ["Female", "Male"], default: "Female" },
  { name: "PhoneService", label: "Phone service", options: ["Yes", "No"], default: "Yes" },
];

// One-click demo profiles. Clicking a preset fills the whole form, which (via the
// debounced effect below) triggers an immediate prediction.
const PRESETS = {
  "Loyal customer": {
    tenure: 65, MonthlyCharges: 25, TotalCharges: 1625, Contract: "Two year",
    InternetService: "DSL", TechSupport: "Yes", OnlineSecurity: "Yes",
    PaymentMethod: "Credit card (automatic)", PaperlessBilling: "No",
    SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", gender: "Female", PhoneService: "Yes",
  },
  "At-risk customer": {
    tenure: 4, MonthlyCharges: 95, TotalCharges: 380, Contract: "Month-to-month",
    InternetService: "Fiber optic", TechSupport: "No", OnlineSecurity: "No",
    PaymentMethod: "Electronic check", PaperlessBilling: "Yes",
    SeniorCitizen: 0, Partner: "No", Dependents: "No", gender: "Male", PhoneService: "Yes",
  },
  "New month-to-month customer": {
    tenure: 1, MonthlyCharges: 70, TotalCharges: 70, Contract: "Month-to-month",
    InternetService: "Fiber optic", TechSupport: "No", OnlineSecurity: "No",
    PaymentMethod: "Electronic check", PaperlessBilling: "Yes",
    SeniorCitizen: 0, Partner: "No", Dependents: "No", gender: "Female", PhoneService: "Yes",
  },
};

// Build the initial form state from each field's default.
const initialForm = () => Object.fromEntries(FIELDS.map((f) => [f.name, f.default]));

export default function Predict() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // The AI insight is loaded separately (on demand), so a slow/failed Gemini call
  // never blocks the instant ML prediction.
  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [insightError, setInsightError] = useState("");

  // Used to ignore out-of-order responses: if the form changes while a request is
  // in flight, only the latest request's result is applied.
  const requestId = useRef(0);

  // --- Reactive prediction -------------------------------------------------
  // Re-run the ML prediction ~400ms after the form stops changing (debounce).
  // This keeps the probability/factors live as you edit, without firing on every
  // keystroke. The "Predict now" button below does the same thing immediately.
  useEffect(() => {
    const myId = ++requestId.current;
    // Any change to the inputs makes a previously-generated AI insight stale.
    setInsight(null);
    setInsightError("");

    const timer = setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const res = await predictCustomer(form);
        if (myId === requestId.current) setResult(res); // ignore stale responses
      } catch (err) {
        if (myId === requestId.current) setError(err.message);
      } finally {
        if (myId === requestId.current) setLoading(false);
      }
    }, 400);

    return () => clearTimeout(timer); // cancel if the form changes again first
  }, [form]);

  function update(name, value, isNumber) {
    setForm((f) => ({ ...f, [name]: isNumber ? Number(value) : value }));
  }

  function applyPreset(name) {
    setForm(PRESETS[name]); // triggers the effect above -> immediate prediction
  }

  // On-demand: ask the backend for the Gemini (or fallback) insight.
  async function generateInsight() {
    setInsightLoading(true);
    setInsightError("");
    setInsight(null);
    try {
      const res = await fetchInsight(form);
      setInsight(res.insight);
    } catch (err) {
      setInsightError(err.message);
    } finally {
      setInsightLoading(false);
    }
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {/* ---- Form ---- */}
      <Card title="Customer details">
        {/* Preset buttons: one click fills the form and predicts immediately. */}
        <div className="mb-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            Quick presets
          </p>
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => applyPreset(name)}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-slate-900 hover:bg-slate-900 hover:text-white"
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {FIELDS.map((field) => (
            <Field key={field.name} field={field} value={form[field.name]} onChange={update} />
          ))}
          <div className="sm:col-span-2">
            {/* The prediction is already reactive; this button forces it now. */}
            <button
              type="button"
              onClick={() => setForm((f) => ({ ...f }))} // re-trigger the effect
              disabled={loading}
              className="w-full rounded-lg bg-slate-900 px-4 py-2.5 font-medium text-white transition hover:bg-slate-700 disabled:opacity-50"
            >
              {loading ? "Scoring…" : "Predict now"}
            </button>
            <p className="mt-2 text-center text-xs text-slate-400">
              Prediction updates automatically as you change fields.
            </p>
          </div>
        </div>
      </Card>

      {/* ---- Result ---- */}
      <div className="space-y-4">
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}
        {!result && !error && (
          <Card>
            <p className="text-slate-500">
              Pick a preset or fill in the customer details. You’ll instantly get a
              churn probability, a colour-coded risk level, the top factors, and a
              recommended retention action. Click <strong>Generate AI insight</strong>{" "}
              for a plain-English explanation.
            </p>
          </Card>
        )}
        {result && (
          <ResultPanel
            result={result}
            insight={insight}
            insightLoading={insightLoading}
            insightError={insightError}
            onGenerateInsight={generateInsight}
          />
        )}
      </div>
    </div>
  );
}

// One labelled input — either a <select> or a number box.
function Field({ field, value, onChange }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block font-medium text-slate-700">{field.label}</span>
      {field.options ? (
        <select
          value={value}
          onChange={(e) => {
            // SeniorCitizen options are numeric; others are strings.
            const numeric = typeof field.options[0] === "object";
            onChange(field.name, e.target.value, numeric);
          }}
          className="w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-slate-500 focus:outline-none"
        >
          {field.options.map((opt) =>
            typeof opt === "object" ? (
              <option key={opt.v} value={opt.v}>{opt.t}</option>
            ) : (
              <option key={opt} value={opt}>{opt}</option>
            )
          )}
        </select>
      ) : (
        <input
          type="number"
          step={field.step || "1"}
          value={value}
          onChange={(e) => onChange(field.name, e.target.value, true)}
          className="w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-slate-500 focus:outline-none"
        />
      )}
    </label>
  );
}

// Shows the scoring result: probability gauge, risk badge, factors, action, and the
// optional AI insight section.
function ResultPanel({ result, insight, insightLoading, insightError, onGenerateInsight }) {
  const pct = (result.churn_probability * 100).toFixed(1);
  const barColor =
    result.risk_level === "High" ? "bg-red-500"
    : result.risk_level === "Medium" ? "bg-amber-500"
    : "bg-green-500";

  return (
    <>
      <Card title="Prediction (from the ML model)">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-4xl font-bold text-slate-900">{pct}%</p>
            <p className="text-sm text-slate-500">probability of churn</p>
          </div>
          <RiskBadge level={result.risk_level} />
        </div>
        <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-100">
          <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
      </Card>

      <Card title="Top factors increasing churn risk">
        {result.top_factors.length === 0 ? (
          <p className="text-sm text-slate-500">No factors are pushing this customer toward churn.</p>
        ) : (
          <ul className="space-y-2">
            {result.top_factors.map((f) => (
              <li key={f.feature} className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-700">
                  {f.label}
                  {f.value !== null && f.value !== undefined && (
                    <span className="text-slate-400"> (value: {String(f.value)})</span>
                  )}
                </span>
                <span className="font-mono text-xs text-slate-500">+{f.contribution.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Recommended retention action">
        <p className="rounded-lg bg-indigo-50 p-3 text-sm font-medium text-indigo-900">
          {result.recommended_action}
        </p>
      </Card>

      {/* AI insight — loaded on demand. The probability above always comes from the
          ML model; Gemini here only explains it and suggests actions in words. */}
      <Card title="AI insight (Gemini)">
        {!insight && !insightLoading && !insightError && (
          <>
            <button
              onClick={onGenerateInsight}
              className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-700"
            >
              ✨ Generate AI insight
            </button>
            <p className="mt-2 text-xs text-slate-400">
              Sends this customer’s ML prediction + factors to Gemini for a plain-English
              explanation. The churn % is never changed by the AI.
            </p>
          </>
        )}
        {insightLoading && <p className="text-sm text-slate-500">Asking Gemini…</p>}
        {insightError && <p className="text-sm text-red-600">{insightError}</p>}
        {insight && <InsightView insight={insight} onRegenerate={onGenerateInsight} />}
      </Card>
    </>
  );
}

// Renders the LLM insight, with a small badge showing whether it came from Gemini
// or the deterministic fallback (so the demo is honest about which path ran).
function InsightView({ insight, onRegenerate }) {
  const isGemini = insight.source === "gemini";
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            isGemini ? "bg-violet-100 text-violet-800" : "bg-slate-100 text-slate-600"
          }`}
        >
          {isGemini ? "Generated by Gemini" : "Fallback (no AI key) — SHAP + rules"}
        </span>
        <button onClick={onRegenerate} className="text-xs text-slate-400 underline hover:text-slate-600">
          regenerate
        </button>
      </div>

      <div>
        <p className="font-semibold text-slate-700">Why this customer is at risk</p>
        <p className="mt-1 text-slate-600">{insight.why_at_risk}</p>
      </div>

      {insight.retention_actions?.length > 0 && (
        <div>
          <p className="font-semibold text-slate-700">Suggested retention actions</p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-slate-600">
            {insight.retention_actions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}

      {insight.outreach_message && (
        <div>
          <p className="font-semibold text-slate-700">Draft outreach message</p>
          <p className="mt-1 rounded-lg bg-slate-50 p-3 italic text-slate-600">
            “{insight.outreach_message}”
          </p>
        </div>
      )}
    </div>
  );
}
