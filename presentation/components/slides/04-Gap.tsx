"use client";
import { motion } from "framer-motion";
import RFRings from "../anim/RFRings";
import { Eyebrow, Reveal, rise } from "../ui";

export default function GapSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1200px]">
        <Eyebrow>the_gap</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Everyone tests the <span className="alert-text">expensive</span> chip
        </motion.h2>

        <div className="mt-8 grid grid-cols-1 items-center gap-10 lg:grid-cols-2">
          <div className="space-y-4">
            <motion.div variants={rise} className="panel p-5 opacity-80">
              <div className="font-mono text-xs text-mute">prior_work</div>
              <div className="mt-1 font-display text-lg text-ink">ESP32-S3 with PSRAM, or a server</div>
              <div className="text-sm text-mute">memory constraint relaxed, or inference offloaded.</div>
            </motion.div>
            <motion.div variants={rise} className="panel p-5 shadow-glow" style={{ borderColor: "rgba(25,230,200,0.55)" }}>
              <div className="font-mono text-xs text-trace">this_work</div>
              <div className="mt-1 font-display text-lg text-trace">Bare classic ESP32 — no PSRAM, ~150 KB SRAM</div>
              <div className="text-sm text-mute">the cheapest, most common, weakest member of the family.</div>
            </motion.div>
            <motion.div variants={rise} className="font-display text-2xl font-medium text-ink md:text-3xl">
              How far can a PSRAM-less ESP32 go?
            </motion.div>
            <motion.div variants={rise} className="font-mono text-xs text-mute">
              ~$3 board · Espressif ranks it the weakest CSI source in the family · never characterized before
            </motion.div>
          </div>

          <motion.div variants={rise} className="relative">
            <div className="panel overflow-hidden p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/figures/fig_board.png" alt="The classic ESP32-WROOM-32 board" className="h-auto w-full rounded-lg" />
              <div className="mt-2 text-center font-mono text-[11px] text-mute">device_under_test // ESP32-WROOM-32 (D0WD-V3)</div>
            </div>
            <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 opacity-60">
              <RFRings className="h-full w-full" />
            </div>
          </motion.div>
        </div>
      </Reveal>
    </div>
  );
}
