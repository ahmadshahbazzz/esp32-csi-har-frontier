"use client";
import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { slides } from "./slides";
import SignalBackground from "./anim/SignalBackground";
import { UcpLogo } from "./Brand";

export default function Deck() {
  const [index, setIndex] = useState(0);
  const [dir, setDir] = useState(1);
  const total = slides.length;

  const go = useCallback(
    (next: number) => {
      setIndex((cur) => {
        const clamped = Math.max(0, Math.min(total - 1, next));
        setDir(clamped >= cur ? 1 : -1);
        return clamped;
      });
    },
    [total]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(index + 1); }
      else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); go(index - 1); }
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(total - 1);
      else if (e.key.toLowerCase() === "f") {
        if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
        else document.exitFullscreen?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, go, total]);

  useEffect(() => {
    const fromHash = () => {
      const i = slides.findIndex((s) => s.id === window.location.hash.replace("#", ""));
      if (i >= 0) setIndex(i);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);
  useEffect(() => {
    history.replaceState(null, "", `#${slides[index].id}`);
  }, [index]);

  const Current = slides[index].Component;
  const sid = String(index + 1).padStart(2, "0");

  return (
    <main className="relative h-[100dvh] w-screen overflow-hidden bg-void">
      <SignalBackground />

      {/* progress */}
      <div className="absolute left-0 top-0 z-40 h-[3px] w-full bg-white/5">
        <motion.div
          className="h-full bg-gradient-to-r from-trace to-amber"
          animate={{ width: `${((index + 1) / total) * 100}%` }}
          transition={{ duration: 0.4 }}
        />
      </div>

      {/* instrument top status bar */}
      <div className="absolute left-0 right-0 top-[3px] z-40 flex items-center justify-between px-5 py-3 font-mono text-[11px] text-mute md:px-9">
        <div className="flex items-center gap-3">
          <UcpLogo size={30} />
          <span className="hidden sm:inline text-trace/80">CSI-HAR // ESP32 deployability</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-trace animate-pulse" />
          <span className="uppercase tracking-wider">{slides[index].label}</span>
          <span className="text-trace">[{sid}/{total}]</span>
        </div>
      </div>

      {/* slides */}
      <div className="relative z-10 h-full w-full">
        <AnimatePresence custom={dir} initial={false}>
          <motion.div
            key={index}
            custom={dir}
            initial={{ opacity: 0, x: dir * 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: dir * -60 }}
            transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
            className="absolute inset-0"
          >
            {/* one-pass scan sweep on entry */}
            <motion.div
              className="scanline z-20"
              initial={{ x: "-120%" }}
              animate={{ x: "120%" }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
            <Current />
          </motion.div>
        </AnimatePresence>
      </div>

      {/* footer identity */}
      <div className="pointer-events-none absolute bottom-4 left-0 right-0 z-40 flex items-center justify-between px-5 font-mono text-[11px] text-mute md:px-9">
        <span>University of Central Punjab</span>
        <span className="hidden md:inline">Muhammad Ahmad · L1F22BSCS0634 · H9</span>
      </div>

      {/* nav */}
      <div className="absolute bottom-3.5 right-5 z-40 flex items-center gap-2 md:right-9">
        <button onClick={() => go(index - 1)} disabled={index === 0}
          className="grid h-9 w-9 place-items-center rounded-md border border-trace/25 bg-panel/60 text-ink transition hover:border-trace hover:text-trace disabled:opacity-25" aria-label="Previous">←</button>
        <button onClick={() => go(index + 1)} disabled={index === total - 1}
          className="grid h-9 w-9 place-items-center rounded-md border border-trace/25 bg-panel/60 text-ink transition hover:border-trace hover:text-trace disabled:opacity-25" aria-label="Next">→</button>
      </div>
    </main>
  );
}
