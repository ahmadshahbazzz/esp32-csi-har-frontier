"use client";
import { motion } from "framer-motion";

/** Concentric RF waves radiating from an emitter point. Pure decoration. */
export default function RFRings({
  className = "",
  color = "#19e6c8",
  count = 5,
}: {
  className?: string;
  color?: string;
  count?: number;
}) {
  return (
    <svg viewBox="0 0 420 420" className={className} aria-hidden>
      <circle cx="210" cy="210" r="10" fill={color} />
      {Array.from({ length: count }).map((_, i) => (
        <motion.circle
          key={i}
          cx="210"
          cy="210"
          r="14"
          fill="none"
          stroke={color}
          strokeWidth="2"
          initial={{ r: 12, opacity: 0 }}
          animate={{ r: [12, 200], opacity: [0, 0.6, 0] }}
          transition={{ duration: 4, delay: (i * 4) / count, repeat: Infinity, ease: "easeOut" }}
        />
      ))}
    </svg>
  );
}
