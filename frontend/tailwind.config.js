export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#070b14", panel: "#0e1626", panel2: "#131f35", line: "#1c2a42", mute: "#8696b3",
        amber: { DEFAULT: "#f2b84b" }, teal: { DEFAULT: "#2ec4b6" }, danger: "#e4572e", good: "#7bd389",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"], mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
};
