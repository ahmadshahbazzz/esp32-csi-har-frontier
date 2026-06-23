"use client";
import { useEffect, useRef } from "react";

/**
 * Live scrolling CSI spectrogram: a time x subcarrier heatmap that scrolls left
 * as new columns arrive, with a teal -> amber -> magenta colormap. Evokes the
 * RuView spectrogram view; here it honestly illustrates Channel State
 * Information amplitude over time. Pure canvas self-blit, no dependencies.
 */
const STOPS: [number, [number, number, number]][] = [
  [0.0, [5, 22, 28]],
  [0.35, [22, 110, 104]],
  [0.6, [25, 230, 200]],
  [0.82, [255, 154, 60]],
  [1.0, [255, 77, 109]],
];

function color(v: number): string {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < STOPS.length; i++) {
    if (v <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1];
      const [b, cb] = STOPS[i];
      const t = (v - a) / (b - a);
      const c = ca.map((x, k) => Math.round(x + (cb[k] - x) * t));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(255,77,109)";
}

export default function CsiSpectrogram({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const ROWS = 40;
    const STEP = 3; // px advanced per tick
    let t = 0;
    let raf = 0;
    let last = 0;

    const seed = () => {
      for (let x = 0; x < canvas.width; x += STEP) { column(x); t += 1; }
    };
    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const nw = Math.max(320, Math.floor(r.width));
      const nh = Math.max(120, Math.floor(r.height));
      if (nw === canvas.width && nh === canvas.height) return;
      canvas.width = nw;
      canvas.height = nh;
      seed(); // fill the whole canvas immediately at the new size
    };

    const column = (x: number) => {
      const W = canvas.width, H = canvas.height;
      const rh = H / ROWS;
      // a slow "activity burst" sweeps through, raising amplitude regionally
      const burst = (Math.sin(t * 0.02) + 1) / 2; // 0..1 over time
      for (let r = 0; r < ROWS; r++) {
        const nr = r / ROWS;
        let v =
          0.4 +
          0.32 * Math.sin(nr * 9 + t * 0.06) +
          0.22 * Math.sin(nr * 22 - t * 0.04) +
          0.18 * Math.sin((nr + burst) * 14 + t * 0.1);
        // localized bright band that drifts (the moving body)
        const band = Math.exp(-Math.pow((nr - (0.3 + 0.4 * burst)) * 6, 2));
        v = v * 0.6 + band * 0.7;
        ctx.fillStyle = color(v);
        ctx.fillRect(x, Math.floor(r * rh), STEP + 1, Math.ceil(rh) + 1);
      }
    };

    resize(); // sizes the canvas and seeds the full heatmap immediately
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      if (ts - last < 33) return; // ~30fps
      last = ts;
      const W = canvas.width, H = canvas.height;
      // scroll left by STEP, draw fresh column on the right
      ctx.drawImage(canvas, STEP, 0, W - STEP, H, 0, 0, W - STEP, H);
      t += 1;
      column(W - STEP);
    };
    if (!reduce) raf = requestAnimationFrame(tick);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={ref} className={className} aria-hidden />;
}
