"use client";
import { useState } from "react";

/**
 * UCP logo framed on a clean white "chip" so the blue/red crest stays legible
 * on the dark deck. Falls back to a text badge only if the file is missing.
 */
export function UcpLogo({ size = 40 }: { size?: number }) {
  const [ok, setOk] = useState(true);
  const pad = Math.round(size * 0.14);
  return (
    <div
      className="grid place-items-center rounded-lg bg-white shadow-[0_0_0_1px_rgba(25,230,200,0.25)]"
      style={{ height: size, width: size, padding: pad }}
    >
      {ok ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src="/ucp.png" alt="University of Central Punjab" className="h-full w-full object-contain" onError={() => setOk(false)} />
      ) : (
        <span className="font-display text-xs font-bold text-[#1b3a8f]">UCP</span>
      )}
    </div>
  );
}

export function Footer() {
  return null;
}
