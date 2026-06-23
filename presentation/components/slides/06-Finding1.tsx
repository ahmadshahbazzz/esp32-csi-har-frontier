"use client";
import { motion } from "framer-motion";
import Counter from "../anim/Counter";
import { Eyebrow, Reveal, rise } from "../ui";

const arenas = [
  { m: "Deep CNN", v: 10 },
  { m: "Deep CNN · CSI-HAR", v: 12 },
  { m: "Transformer", v: 24 },
  { m: "Transformer · CSI-HAR", v: 54 },
];

export default function Finding1Slide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1220px]">
        <Eyebrow>finding_01</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Deployment is <span className="gradient-text">not</span> the wall
        </motion.h2>
        <motion.p variants={rise} className="mt-3 max-w-3xl text-lg text-mute">
          We flashed every convertible model to the board. A full deep CNN and a Transformer both
          run on the bare classic ESP32. The trick: weights sit in the 4 MB flash, and only the
          activations (the tensor arena) need SRAM, so they land far below the ~150 KB budget. There
          is no general memory wall.
        </motion.p>

        <div className="mt-7 grid grid-cols-1 items-center gap-7 lg:grid-cols-2">
          <motion.div variants={rise} className="grid grid-cols-2 gap-3">
            {arenas.map((d) => (
              <div key={d.m} className="panel p-4 text-center">
                <div className="font-display text-4xl font-bold text-trace">
                  <Counter to={d.v} />
                  <span className="text-lg text-mute"> kB</span>
                </div>
                <div className="mt-1 font-mono text-[11px] text-mute">{d.m}</div>
              </div>
            ))}
            <div className="col-span-2 rounded-xl border border-amber/30 bg-amber/10 px-4 py-3 text-center font-mono text-sm text-ink">
              available budget on the bare chip ≈ <span className="font-bold text-amber">150 kB</span>
            </div>
          </motion.div>

          <motion.div variants={rise} className="panel overflow-hidden p-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/figures/fig_memory.png" alt="Measured on-device tensor arena vs the budget" className="h-auto w-full rounded-lg bg-white" />
          </motion.div>
        </div>
      </Reveal>
    </div>
  );
}
