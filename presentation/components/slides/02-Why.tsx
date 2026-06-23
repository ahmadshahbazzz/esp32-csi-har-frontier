"use client";
import { motion } from "framer-motion";
import RFRings from "../anim/RFRings";
import { Eyebrow, Reveal, rise } from "../ui";

const points = [
  { k: "no_camera", t: "No camera", d: "Works in the dark and in bedrooms or bathrooms. Nothing is ever filmed." },
  { k: "no_wearable", t: "No wearable", d: "Nothing to charge, clip on, or remember. The person carries nothing." },
  { k: "scales_cheaply", t: "Already deployed", d: "Re-uses the WiFi radios already in every home, ward, and office." },
];

export default function WhySlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1180px]">
        <Eyebrow>motivation</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Sensing people <span className="gradient-text">without watching them</span>
        </motion.h2>
        <motion.p variants={rise} className="mt-4 max-w-3xl text-lg text-mute md:text-xl">
          A moving body bends and scatters the WiFi signal already in the room. A receiver reads
          those tiny changes and recognizes what the person is doing. For elderly fall-detection or
          patient monitoring, this camera-free, wearable-free privacy is the whole point.
        </motion.p>

        <div className="mt-10 grid grid-cols-1 gap-5 md:grid-cols-3">
          {points.map((p) => (
            <motion.div key={p.k} variants={rise} className="panel p-6">
              <div className="font-mono text-xs text-trace">{p.k}</div>
              <div className="mt-2 font-display text-2xl font-medium text-ink">{p.t}</div>
              <div className="mt-2 text-sm text-mute">{p.d}</div>
            </motion.div>
          ))}
        </div>
      </Reveal>

      <div className="pointer-events-none absolute -right-16 bottom-0 hidden h-[420px] w-[420px] opacity-40 lg:block">
        <RFRings className="h-full w-full" color="#ff9a3c" />
      </div>
    </div>
  );
}
