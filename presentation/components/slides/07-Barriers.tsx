"use client";
import { motion } from "framer-motion";
import { Eyebrow, Reveal, rise } from "../ui";

const rows = [
  { m: "CNN", conv: "yes", dev: "fits", ok: true },
  { m: "Transformer", conv: "yes", dev: "fits", ok: true },
  { m: "Chebyshev-KAN", conv: "yes", dev: "allocation fails", ok: false },
  { m: "BiGRU", conv: "no", dev: "not convertible", ok: false },
  { m: "Lightweight-SSM", conv: "no", dev: "not convertible", ok: false },
];

export default function BarriersSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1100px]">
        <Eyebrow>finding_01b // what_blocks_the_rest</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          Two narrow walls, not a memory wall
        </motion.h2>
        <motion.p variants={rise} className="mt-3 max-w-3xl text-lg text-mute">
          What blocks the rest is <span className="text-ink">convertibility</span>: the recurrent and
          state-space models compile to fused operators that TensorFlow Lite Micro does not support,
          so they never reach the chip. The KAN converts but hits a{" "}
          <span className="text-ink">runtime allocation failure</span>. Neither is a memory-size wall.
        </motion.p>

        <motion.div variants={rise} className="panel mt-8 overflow-hidden">
          <table className="w-full text-left">
            <thead className="font-mono text-xs uppercase tracking-wider text-mute">
              <tr className="border-b border-trace/15">
                <th className="px-5 py-3">deep_model</th>
                <th className="px-5 py-3">converts_int8</th>
                <th className="px-5 py-3">on_bare_esp32</th>
              </tr>
            </thead>
            <tbody className="text-lg">
              {rows.map((r) => (
                <tr key={r.m} className="border-b border-white/5 last:border-0">
                  <td className="px-5 py-3 font-display font-medium text-ink">{r.m}</td>
                  <td className="px-5 py-3 font-mono text-sm text-mute">{r.conv}</td>
                  <td className={`px-5 py-3 font-mono text-sm font-bold ${r.ok ? "text-trace" : "text-alert"}`}>
                    {r.ok ? "✓ " : "✗ "}{r.dev}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </Reveal>
    </div>
  );
}
