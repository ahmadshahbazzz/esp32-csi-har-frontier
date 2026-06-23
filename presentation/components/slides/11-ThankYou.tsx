"use client";
import { motion } from "framer-motion";
import ScopeTrace from "../anim/ScopeTrace";
import { Eyebrow, Reveal, rise } from "../ui";

export default function ThankYouSlide() {
  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/img/cyber.jpg" alt="" className="absolute inset-0 h-full w-full object-cover opacity-[0.18]" />
      <div className="absolute inset-0 bg-gradient-to-t from-void via-void/85 to-void/55" />

      <div className="slide-pad relative z-10 flex h-full flex-col justify-center">
        <Reveal className="max-w-[1000px]">
          <Eyebrow>end_of_transmission</Eyebrow>
          <motion.h2 variants={rise} className="mt-5 font-display text-5xl font-bold md:text-7xl">
            Thank you
          </motion.h2>
          <motion.p variants={rise} className="mt-4 max-w-2xl text-xl text-mute">
            Deployment is solved on the cheapest WiFi microcontroller. The next frontier is helping
            these models recognize a person they have never seen.
          </motion.p>

          <motion.div variants={rise} className="panel mt-9 inline-block px-7 py-5">
            <div className="font-display text-xl font-medium text-ink">Muhammad Ahmad</div>
            <div className="font-mono text-sm text-trace">L1F22BSCS0634 · Section H9</div>
            <div className="mt-1 font-mono text-xs text-mute">University of Central Punjab</div>
          </motion.div>

          <motion.div variants={rise} className="mt-7 font-display text-lg text-amber">Questions?</motion.div>
        </Reveal>
      </div>

      <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-0 opacity-50">
        <ScopeTrace className="h-[22vh] w-full" />
      </div>
    </div>
  );
}
