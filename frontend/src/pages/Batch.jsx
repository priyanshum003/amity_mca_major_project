import { useState } from "react";
import Card from "../components/Card.jsx";
import { predictBatch } from "../api.js";

// Batch screen: upload a CSV of customers, the backend scores every row and
// returns the same CSV with churn_probability, risk_level and top_factor added.
// We show a preview table and offer a download of the scored file.
export default function Batch() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);   // { headers, rows }
  const [downloadUrl, setDownloadUrl] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onScore() {
    if (!file) return;
    setLoading(true);
    setError("");
    setPreview(null);
    setDownloadUrl("");
    try {
      const blob = await predictBatch(file);
      const text = await blob.text();
      setPreview(parseCsvPreview(text, 25));
      setDownloadUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card title="Upload a customer CSV">
        <p className="mb-4 text-sm text-slate-500">
          Upload a CSV with the Telco customer columns (the same format as
          <code> data/telco_churn.csv</code>). Missing columns are filled with
          sensible defaults. Each row gets a churn probability, risk level and its
          top contributing factor.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0] || null)}
            className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:text-white hover:file:bg-slate-700"
          />
          <button
            onClick={onScore}
            disabled={!file || loading}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:opacity-50"
          >
            {loading ? "Scoring…" : "Score customers"}
          </button>
          {downloadUrl && (
            <a
              href={downloadUrl}
              download="scored_customers.csv"
              className="rounded-lg border border-green-600 px-4 py-2 text-sm font-medium text-green-700 transition hover:bg-green-50"
            >
              ⬇ Download scored CSV
            </a>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </Card>

      {preview && (
        <Card title={`Scored preview (first ${preview.rows.length} rows)`}>
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  {preview.headers.map((h) => (
                    <th key={h} className="whitespace-nowrap px-2 py-2 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i} className="border-b border-slate-100">
                    {row.map((cell, j) => (
                      <td key={j} className="whitespace-nowrap px-2 py-1.5">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Preview only — use the download button for the full scored file.
          </p>
        </Card>
      )}
    </div>
  );
}

// Minimal CSV parser just for the preview table (handles simple quoted fields).
// The real parsing happens server-side with pandas; this is display-only.
function parseCsvPreview(text, maxRows) {
  const lines = text.trim().split(/\r?\n/);
  const split = (line) => {
    const out = [];
    let cur = "", inQuotes = false;
    for (const ch of line) {
      if (ch === '"') inQuotes = !inQuotes;
      else if (ch === "," && !inQuotes) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  };
  const headers = split(lines[0]);
  const rows = lines.slice(1, maxRows + 1).map(split);
  return { headers, rows };
}
