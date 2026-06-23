"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import type { Scenario } from "../anim/Observatory3D";
import { Eyebrow } from "../ui";

const Observatory3D = dynamic(() => import("../anim/Observatory3D"), { ssr: false });

const scenarios: { id: Scenario; label: string; activity: string }[] = [
  { id: "empty", label: "Empty room", activity: "no presence" },
  { id: "stand", label: "Standing", activity: "standing" },
  { id: "walk", label: "Walking", activity: "walking" },
  { id: "fall", label: "Fall", activity: "fall detected" },
];

export default function DemoSlide() {
  const [scenario, setScenario] = useState<Scenario>("walk");
  const [bloom, setBloom] = useState(1.15);
  const [waveSpeed, setWaveSpeed] = useState(1);
  const [autoRotate, setAutoRotate] = useState(false);
  const active = scenarios.find((s) => s.id === scenario)!;
  const present = scenario !== "empty";

  return (
    <div className="relative h-full w-full overflow-hidden">
      <Observatory3D className="absolute inset-0 h-full w-full" interactive scenario={scenario} bloom={bloom} waveSpeed={waveSpeed} autoRotate={autoRotate} />

      {/* header */}
      <div className="pointer-events-none absolute left-0 top-0 z-10 px-6 pt-16 md:px-12 md:pt-20">
        <Eyebrow>live_demo // wifi_presence_sensing</Eyebrow>
        <h2 className="mt-3 font-display text-2xl font-bold md:text-4xl">An interactive look at the idea</h2>
        <p className="mt-2 max-w-md font-mono text-xs text-mute">drag to orbit · scroll to zoom · illustrative visualization</p>
      </div>

      {/* status readout (RuView-style) */}
      <div className="absolute right-6 top-16 z-10 w-[190px] md:right-12 md:top-20">
       <div className="panel p-4 font-mono text-xs">
        <div className="text-mute">status</div>
        <div className="mt-2 flex items-center justify-between">
          <span>presence</span>
          <span className={present ? "text-trace" : "text-mute"}>{present ? "YES" : "—"}</span>
        </div>
        <div className="mt-1 flex items-center justify-between">
          <span>activity</span>
          <span className={scenario === "fall" ? "text-alert" : "text-ink"}>{active.activity}</span>
        </div>
        <div className="mt-1 flex items-center justify-between">
          <span>WiFi RSSI</span>
          <span className="text-ink">-{present ? 48 : 71} dBm</span>
        </div>
        {scenario === "fall" && (
          <motion.div animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 0.9, repeat: Infinity }} className="mt-3 rounded-md bg-alert/20 px-2 py-1.5 text-center font-bold text-alert">
            ⚠ FALL DETECTED
          </motion.div>
        )}
       </div>
      </div>

      {/* control panel */}
      <div className="absolute bottom-20 left-1/2 z-10 w-[min(92vw,880px)] -translate-x-1/2">
       <div className="panel p-4">
        <div className="flex flex-wrap items-center gap-2">
          {scenarios.map((s) => (
            <button
              key={s.id}
              onClick={() => setScenario(s.id)}
              className={`rounded-md border px-4 py-2 font-mono text-sm transition ${
                scenario === s.id ? "border-trace bg-trace/15 text-trace" : "border-white/15 text-mute hover:border-trace/50 hover:text-ink"
              }`}
            >
              {s.label}
            </button>
          ))}
          <div className="ml-auto flex flex-wrap items-center gap-5">
            <label className="flex items-center gap-2 font-mono text-xs text-mute">
              glow
              <input type="range" min={0} max={2.5} step={0.05} value={bloom} onChange={(e) => setBloom(+e.target.value)} className="accent-trace" />
            </label>
            <label className="flex items-center gap-2 font-mono text-xs text-mute">
              wave speed
              <input type="range" min={0.2} max={3} step={0.1} value={waveSpeed} onChange={(e) => setWaveSpeed(+e.target.value)} className="accent-trace" />
            </label>
            <button
              onClick={() => setAutoRotate((v) => !v)}
              className={`rounded-md border px-3 py-1.5 font-mono text-xs transition ${
                autoRotate ? "border-amber bg-amber/15 text-amber" : "border-white/15 text-mute hover:text-ink"
              }`}
            >
              auto-orbit {autoRotate ? "on" : "off"}
            </button>
          </div>
        </div>
       </div>
      </div>
    </div>
  );
}
