"use client";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import { UcpLogo } from "../Brand";
import { Eyebrow, Reveal, rise } from "../ui";

const Observatory3D = dynamic(() => import("../anim/Observatory3D"), { ssr: false });

export default function TitleSlide() {
  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* 3D WiFi observatory scene */}
      <Observatory3D className="absolute inset-0 h-full w-full" />
      {/* readability gradient over the left half */}
      <div className="absolute inset-0 bg-gradient-to-r from-void via-void/80 to-transparent" />
      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-void to-transparent" />

      <div className="slide-pad relative z-10 flex h-full flex-col justify-center">
        <Reveal className="max-w-[760px]">
          <motion.div variants={rise} className="mb-6 flex items-center gap-3">
            <UcpLogo size={64} />
            <div className="font-mono text-sm leading-tight text-ink">
              University of Central Punjab
              <div className="text-xs text-mute">Faculty of Information Technology</div>
            </div>
          </motion.div>
          <Eyebrow>tinyml // wifi_sensing // on_device_ai</Eyebrow>

          <motion.h1 variants={rise} className="mt-6 font-display text-[2.5rem] font-bold leading-[1.04] tracking-tight md:text-[4.4rem]">
            How far can a<br />
            <span className="gradient-text">PSRAM-less ESP32</span> go?
          </motion.h1>

          <motion.p variants={rise} className="mt-5 max-w-xl text-lg text-mute md:text-2xl">
            A TinyML deployment study for WiFi CSI human activity recognition
            <span className="text-ink"> on a $5 microcontroller.</span>
          </motion.p>

          <motion.div variants={rise} className="mt-9">
            <div className="font-display text-xl font-medium text-ink">Muhammad Ahmad</div>
            <div className="font-mono text-sm text-trace">L1F22BSCS0634 · Section H9</div>
          </motion.div>

          <motion.div variants={rise} className="mt-8 font-mono text-xs text-mute">
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 text-ink">→</kbd> next ·{" "}
            <kbd className="rounded bg-white/10 px-1.5 py-0.5 text-ink">F</kbd> fullscreen
          </motion.div>
        </Reveal>
      </div>
    </div>
  );
}
