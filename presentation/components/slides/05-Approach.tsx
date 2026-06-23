"use client";
import { motion } from "framer-motion";
import { Eyebrow, Reveal, rise } from "../ui";

const tiers = [
  { id: "tier_0", name: "Classical", ex: "Decision Tree · Random Forest · MLP", c: "#19e6c8" },
  { id: "tier_1", name: "Tiny NN", ex: "Tiny-CNN width sweep · tiny MLP", c: "#ff9a3c" },
  { id: "tier_2", name: "Deep", ex: "CNN · BiGRU · Transformer · KAN · SSM", c: "#7aa6ff" },
];
const pipe = ["train + int8 quantize", "convert to TFLite Micro", "flash to ESP32", "measure fit · latency · RAM"];

export default function ApproachSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1220px]">
        <Eyebrow>approach</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          A <span className="gradient-text">deployability frontier</span>
        </motion.h2>
        <motion.p variants={rise} className="mt-3 max-w-3xl text-lg text-mute">
          Three model tiers across two public datasets (UT-HAR and CSI-HAR, 7 activities each), with
          one honest question per model: does it actually fit and run on the bare chip? CSI-HAR is
          tested leave-one-user-out, so accuracy reflects a genuinely unseen person.
        </motion.p>

        <div className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-3">
          {tiers.map((t) => (
            <motion.div key={t.id} variants={rise} className="panel p-6">
              <div className="flex items-center gap-2 font-mono text-xs" style={{ color: t.c }}>
                <span className="h-2 w-2 rounded-full" style={{ background: t.c }} />
                {t.id}
              </div>
              <div className="mt-2 font-display text-2xl font-medium text-ink">{t.name}</div>
              <div className="mt-1 text-sm text-mute">{t.ex}</div>
            </motion.div>
          ))}
        </div>

        <motion.div variants={rise} className="mt-8 flex flex-wrap items-center gap-x-2 gap-y-3 font-mono text-sm">
          {pipe.map((p, i) => (
            <span key={p} className="flex items-center gap-2">
              <span className="rounded-md border border-trace/25 bg-panel/60 px-4 py-2 text-ink">{p}</span>
              {i < pipe.length - 1 && <span className="text-amber">→</span>}
            </span>
          ))}
        </motion.div>
      </Reveal>
    </div>
  );
}
