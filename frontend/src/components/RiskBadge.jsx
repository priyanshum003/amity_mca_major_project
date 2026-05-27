// A coloured pill that shows a Low / Medium / High risk level.
// Colours match the Tailwind `risk` palette defined in tailwind.config.js.
const STYLES = {
  Low: "bg-green-100 text-green-800 ring-green-600/20",
  Medium: "bg-amber-100 text-amber-800 ring-amber-600/20",
  High: "bg-red-100 text-red-800 ring-red-600/20",
};

export default function RiskBadge({ level }) {
  const style = STYLES[level] || "bg-slate-100 text-slate-700 ring-slate-500/20";
  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ring-inset ${style}`}>
      {level} risk
    </span>
  );
}
