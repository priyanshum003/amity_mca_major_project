// labels.js
// ---------
// Turns the model's encoded feature names into human-readable labels for the
// charts. Mirrors the backend's schema.readable_label so the dashboard and the
// per-prediction factors read consistently.
//
//   "num__tenure"                  -> "Tenure (months)"
//   "cat__Contract_Month-to-month" -> "Contract: Month-to-month"

const COLUMN_LABELS = {
  tenure: "Tenure (months)",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
  Contract: "Contract",
  InternetService: "Internet service",
  OnlineSecurity: "Online security",
  TechSupport: "Tech support",
  PaymentMethod: "Payment method",
  PaperlessBilling: "Paperless billing",
  SeniorCitizen: "Senior citizen",
};

// Known categorical columns, longest first so "OnlineSecurity" matches before
// a shorter accidental prefix would.
const CATEGORICAL = [
  "PaymentMethod", "PaperlessBilling", "DeviceProtection", "StreamingMovies",
  "OnlineSecurity", "InternetService", "MultipleLines", "SeniorCitizen",
  "StreamingTV", "PhoneService", "TechSupport", "OnlineBackup", "Dependents",
  "Contract", "Partner", "gender",
].sort((a, b) => b.length - a.length);

export function prettyDriver(encodedName) {
  const name = encodedName.includes("__") ? encodedName.split("__")[1] : encodedName;

  // Numeric / no value suffix.
  if (COLUMN_LABELS[name] && !name.includes("_")) return COLUMN_LABELS[name];
  if (["tenure", "MonthlyCharges", "TotalCharges"].includes(name)) {
    return COLUMN_LABELS[name] || name;
  }

  for (const col of CATEGORICAL) {
    if (name === col) return COLUMN_LABELS[col] || col;
    if (name.startsWith(col + "_")) {
      const value = name.slice(col.length + 1);
      return `${COLUMN_LABELS[col] || col}: ${value}`;
    }
  }
  return name;
}
