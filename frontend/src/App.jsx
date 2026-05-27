import { useState } from "react";
import Dashboard from "./pages/Dashboard.jsx";
import Predict from "./pages/Predict.jsx";
import Batch from "./pages/Batch.jsx";

// The three screens of the app. We use a tiny tab state instead of a router —
// fewer dependencies and easy to explain: clicking a tab just swaps which
// page component is rendered.
const TABS = [
  { id: "dashboard", label: "Dashboard", component: Dashboard },
  { id: "predict", label: "Predict a customer", component: Predict },
  { id: "batch", label: "Batch upload", component: Batch },
];

export default function App() {
  const [active, setActive] = useState("dashboard");
  const ActivePage = TABS.find((t) => t.id === active).component;

  return (
    <div className="min-h-screen">
      {/* Header + tab navigation */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-4">
          <h1 className="text-xl font-bold text-slate-900">
            Churn Retention Dashboard
          </h1>
          <p className="text-sm text-slate-500">
            Explainable ML for retention marketing — Telco customer churn
          </p>
          <nav className="mt-4 flex gap-2">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActive(tab.id)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                  active === tab.id
                    ? "bg-slate-900 text-white"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <ActivePage />
      </main>
    </div>
  );
}
