"use client";
import { motion } from "framer-motion";

/**
 * Signal-drop visualization for the generalization gap: a trace holds a high
 * plateau (random split, ~96%) then plummets to a low plateau (unseen person,
 * ~63%). The crash is the whole point of the paper, rendered as an instrument
 * event.
 */
export default function GapDrop({ className = "" }: { className?: string }) {
  const W = 640;
  // y: 40 = 100%, 260 = 0%
  const y = (pct: number) => 260 - (pct / 100) * 220;
  const high = y(96.4);
  const low = y(63.1);

  // wobbly high plateau, sharp drop near x=330, wobbly low plateau
  const pts: string[] = [];
  for (let x = 0; x <= W; x += 12) {
    let base = x < 330 ? high : low;
    if (x >= 318 && x < 342) base = high + ((low - high) * (x - 318)) / 24; // the crash ramp
    const wob = Math.sin(x / 18) * (x < 330 ? 4 : 6);
    pts.push(`${x},${(base + wob).toFixed(1)}`);
  }
  const d = "M " + pts.join(" L ");

  return (
    <svg viewBox="0 0 640 300" className={className} aria-hidden>
      <defs>
        <linearGradient id="gap-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#19e6c8" />
          <stop offset="50%" stopColor="#19e6c8" />
          <stop offset="58%" stopColor="#ff4d6d" />
          <stop offset="100%" stopColor="#ff4d6d" />
        </linearGradient>
      </defs>

      {/* reference rails */}
      <line x1="0" y1={high} x2="640" y2={high} stroke="rgba(25,230,200,0.35)" strokeDasharray="2 6" />
      <line x1="0" y1={low} x2="640" y2={low} stroke="rgba(255,77,109,0.4)" strokeDasharray="2 6" />
      <text x="8" y={high - 8} fontFamily="JetBrains Mono" fontSize="13" fill="#19e6c8">random split 96%</text>
      <text x="8" y={low + 20} fontFamily="JetBrains Mono" fontSize="13" fill="#ff4d6d">unseen person 63%</text>

      {/* the trace */}
      <motion.path
        d={d}
        fill="none"
        stroke="url(#gap-grad)"
        strokeWidth="3"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 2.2, ease: "easeInOut" }}
      />

      {/* drop marker */}
      <motion.g
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 0.5 }}
      >
        <line x1="365" y1={high} x2="365" y2={low} stroke="#ff4d6d" strokeWidth="1.5" strokeDasharray="3 3" />
        <text x="378" y={(high + low) / 2 + 5} fontFamily="Space Grotesk" fontSize="22" fontWeight="700" fill="#ff4d6d">
          -29 pts
        </text>
      </motion.g>
    </svg>
  );
}
