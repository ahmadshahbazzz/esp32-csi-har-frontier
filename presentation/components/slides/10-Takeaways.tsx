"use client";
import { motion } from "framer-motion";
import { Eyebrow, Reveal, rise } from "../ui";

const items = [
  { id: "01", k: "Deployment is solved", v: "On the cheapest classic ESP32, even deep CNN and Transformer models fit and run. The chip is no longer the limit." },
  { id: "02", k: "Generalization is the wall", v: "Accuracy drops ~29 points on an unseen person. That, not memory, is the open problem for WiFi sensing here." },
  { id: "03", k: "Report it honestly", v: "Random-split numbers overstate accuracy. Subject-independent evaluation is what a deployed device faces." },
  { id: "04", k: "Small is smart", v: "Tiny CNNs and classical learners match deep accuracy at a fraction of the size and latency." },
];

export default function TakeawaysSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1160px]">
        <Eyebrow>takeaways</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          What this means
        </motion.h2>

        <div className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
          {items.map((it) => (
            <motion.div key={it.id} variants={rise} className="panel p-6">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-sm text-trace">{it.id}</span>
                <span className="font-display text-xl font-medium text-ink">{it.k}</span>
              </div>
              <div className="mt-2 text-sm text-mute">{it.v}</div>
            </motion.div>
          ))}
        </div>
      </Reveal>
    </div>
  );
}
