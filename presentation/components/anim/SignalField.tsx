"use client";
import { useEffect, useRef } from "react";

/**
 * Glowing "signal field": flowing, interfering wave lines with additive bloom,
 * inspired by the RuView observatory's WiFi-wave / signal-field aesthetic.
 * Pure canvas, no dependencies. Ambient background element.
 */
export default function SignalField({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let w = 0, h = 0, dpr = Math.min(2, window.devicePixelRatio || 1);

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = r.width; h = r.height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const lines = [
      { color: "25,230,200", amp: 0.10, freq: 1.6, speed: 0.18, off: 0.0 },
      { color: "25,230,200", amp: 0.07, freq: 2.4, speed: 0.12, off: 1.2 },
      { color: "255,154,60", amp: 0.06, freq: 3.1, speed: 0.10, off: 2.6 },
      { color: "122,166,255", amp: 0.05, freq: 4.2, speed: 0.08, off: 4.1 },
    ];

    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      const time = t * 0.001;
      for (const l of lines) {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 6) {
          const nx = x / w;
          const y =
            h * 0.5 +
            Math.sin(nx * Math.PI * l.freq + time * l.speed * 6 + l.off) * h * l.amp +
            Math.sin(nx * Math.PI * l.freq * 2.3 + time * l.speed * 3) * h * l.amp * 0.4;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(${l.color},0.5)`;
        ctx.lineWidth = 1.6;
        ctx.shadowBlur = 18;
        ctx.shadowColor = `rgba(${l.color},0.7)`;
        ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
      ctx.shadowBlur = 0;
      if (!reduce) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={ref} className={className} aria-hidden />;
}
