"use client";
import { motion } from "framer-motion";

/**
 * Oscilloscope-style CSI trace: stacked subcarrier lines that self-draw, then
 * gently morph forever, evoking a live channel-state readout. The deck's
 * recurring signature motif.
 */
function path(seed: number, amp: number, w = 640, h = 220): string {
  const mid = h / 2;
  const pts: string[] = [];
  for (let x = 0; x <= w; x += 16) {
    const y =
      mid +
      Math.sin((x / w) * Math.PI * 4 + seed) * amp +
      Math.sin((x / w) * Math.PI * 11 + seed * 1.7) * amp * 0.35;
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return "M " + pts.join(" L ");
}

export default function ScopeTrace({ className = "" }: { className?: string }) {
  const ch = [
    { c: "#19e6c8", a: 36, s: 0 },
    { c: "#ff9a3c", a: 26, s: 1.3 },
    { c: "#7aa6ff", a: 18, s: 2.7 },
  ];
  return (
    <svg viewBox="0 0 640 220" className={className} aria-hidden>
      {/* baseline */}
      <line x1="0" y1="110" x2="640" y2="110" stroke="rgba(25,230,200,0.18)" strokeDasharray="3 6" />
      {ch.map((l, i) => (
        <motion.path
          key={i}
          fill="none"
          stroke={l.c}
          strokeWidth="2.4"
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{
            pathLength: 1,
            opacity: 0.92,
            d: [path(l.s, l.a), path(l.s + 2, l.a * 0.6), path(l.s + 4, l.a), path(l.s, l.a)],
          }}
          transition={{
            pathLength: { duration: 1.1, delay: i * 0.18, ease: "easeInOut" },
            opacity: { duration: 0.5, delay: i * 0.18 },
            d: { duration: 7, repeat: Infinity, ease: "easeInOut", delay: 1.2 },
          }}
        />
      ))}
    </svg>
  );
}
