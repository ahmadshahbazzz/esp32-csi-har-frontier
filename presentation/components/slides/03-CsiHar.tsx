"use client";
import { motion } from "framer-motion";
import CsiSpectrogram from "../anim/CsiSpectrogram";
import ScopeTrace from "../anim/ScopeTrace";
import { Eyebrow, Reveal, rise } from "../ui";

const steps = [
  { n: "01", t: "WiFi fills the room", d: "The router transmits packets continuously." },
  { n: "02", t: "A body perturbs it", d: "Walking, sitting, or falling scatters the signal differently." },
  { n: "03", t: "CSI records the change", d: "Channel State Information = amplitude per subcarrier, over time." },
  { n: "04", t: "A model reads the action", d: "A classifier maps the CSI pattern to walk, run, sit, stand, fall, bend, or lie down." },
];

export default function CsiHarSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1200px]">
        <Eyebrow>background // what_is_csi_har</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Reading activity from a <span className="gradient-text">radio signal</span>
        </motion.h2>

        <div className="mt-8 grid grid-cols-1 items-center gap-8 lg:grid-cols-2">
          <motion.div variants={rise} className="panel overflow-hidden p-5">
            <div className="flex items-center justify-between font-mono text-xs text-mute">
              <span>CSI spectrogram // 52 subcarriers</span>
              <span className="flex items-center gap-1.5 text-trace"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-trace" />live</span>
            </div>
            <div className="mt-3 overflow-hidden rounded-lg">
              <CsiSpectrogram className="block h-[150px] w-full" />
            </div>
            <ScopeTrace className="mt-2 h-[90px] w-full" />
            <div className="mt-1 font-mono text-[11px] text-mute">amplitude per subcarrier over time; the bright band is the moving body</div>
          </motion.div>

          <div className="space-y-3">
            {steps.map((s) => (
              <motion.div key={s.n} variants={rise} className="flex items-start gap-4">
                <div className="mt-0.5 font-mono text-lg font-bold text-trace">{s.n}</div>
                <div>
                  <div className="font-display text-xl font-medium text-ink">{s.t}</div>
                  <div className="text-sm text-mute">{s.d}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Reveal>
    </div>
  );
}
