/** Tailwind CSS configuration. We scan index.html and every file under src/
 *  for class names so unused styles are purged from the production build. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Risk-level palette reused across badges and charts.
        risk: {
          low: "#16a34a",    // green-600
          medium: "#d97706", // amber-600
          high: "#dc2626",   // red-600
        },
      },
    },
  },
  plugins: [],
};
