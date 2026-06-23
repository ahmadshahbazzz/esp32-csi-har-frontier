"use client";
import { motion } from "framer-motion";
import Counter from "../anim/Counter";
import { Eyebrow, Reveal, rise } from "../ui";

const stats = [
  { v: 96.7, dec: 1, suf: "%", l: "tiny CNN accuracy · UT-HAR" },
  { v: 8, dec: 0, suf: "-13 kB", l: "tiny-model RAM on device" },
  { v: 14, dec: 0, suf: "-117 ms", l: "inference latency" },
  { v: 45, dec: 0, suf: " us", l: "Random Forest inference" },
];

export default function ResultsSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1220px]">
        <Eyebrow>results // measured_on_hardware</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Small models, <span className="gradient-text">real numbers</span>
        </motion.h2>

        <div className="mt-7 grid grid-cols-2 gap-3 md:grid-cols-4">
          {stats.map((s) => (
            <motion.div key={s.l} variants={rise} className="panel p-5 text-center">
              <div className="font-display text-3xl font-bold text-trace md:text-4xl">
                <Counter to={s.v} decimals={s.dec} suffix={s.suf} />
              </div>
              <div className="mt-1 font-mono text-[11px] text-mute">{s.l}</div>
            </motion.div>
          ))}
        </div>

        <motion.div variants={rise} className="mt-6 panel overflow-hidden p-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/figures/fig_accuracy.png" alt="Accuracy across tiers and datasets" className="mx-auto h-[34vh] w-auto rounded-lg bg-white" />
          <div className="mt-2 text-center font-mono text-[11px] text-mute">
            UT-HAR is saturated; CSI-HAR (leave-one-user-out) is the honest cross-subject test
          </div>
        </motion.div>
      </Reveal>
    </div>
  );
}
