import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#04141a",
        panel: "#082028",
        panel2: "#0b2a33",
        trace: "#19e6c8",
        amber: "#ff9a3c",
        alert: "#ff4d6d",
        ink: "#d7efec",
        mute: "#74939a",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      boxShadow: {
        glow: "0 0 60px -12px rgba(25,230,200,0.5)",
        alert: "0 0 50px -10px rgba(255,77,109,0.55)",
      },
    },
  },
  plugins: [],
};
export default config;
