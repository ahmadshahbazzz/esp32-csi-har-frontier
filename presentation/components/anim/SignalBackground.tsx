"use client";
import { motion } from "framer-motion";

/** Instrument backdrop: RF dot-grid + two slow drifting signal glows. */
export default function SignalBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 rf-grid" />
      <motion.div
        className="absolute -left-48 -top-48 h-[560px] w-[560px] rounded-full"
        style={{ background: "radial-gradient(circle, rgba(25,230,200,0.16), transparent 70%)" }}
        animate={{ x: [0, 70, 0], y: [0, 40, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute -right-48 bottom-[-180px] h-[600px] w-[600px] rounded-full"
        style={{ background: "radial-gradient(circle, rgba(255,154,60,0.12), transparent 70%)" }}
        animate={{ x: [0, -55, 0], y: [0, -34, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}
