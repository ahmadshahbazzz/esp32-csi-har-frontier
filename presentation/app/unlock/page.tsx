"use client";
import { useState } from "react";

export default function Unlock() {
  const [pw, setPw] = useState("");
  const [err, setErr] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(false);
    const r = await fetch("/api/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pw }),
    });
    setBusy(false);
    if (r.ok) window.location.href = "/";
    else setErr(true);
  };

  return (
    <main className="flex h-[100dvh] w-screen items-center justify-center bg-void">
      <div className="pointer-events-none absolute inset-0 rf-grid" />
      <form onSubmit={submit} className="panel relative z-10 w-[min(92vw,380px)] p-7">
        <div className="eyebrow">{">"} restricted // enter_password</div>
        <h1 className="mt-3 font-display text-2xl font-bold text-ink">
          WiFi CSI HAR <span className="gradient-text">presentation</span>
        </h1>
        <p className="mt-1 font-mono text-xs text-mute">University of Central Punjab</p>
        <input
          type="password"
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          autoFocus
          placeholder="password"
          className="mt-5 w-full rounded-md border border-trace/25 bg-black/30 px-4 py-3 font-mono text-ink outline-none focus:border-trace"
        />
        {err && <div className="mt-2 font-mono text-xs text-alert">incorrect password</div>}
        <button
          type="submit"
          disabled={busy}
          className="mt-4 w-full rounded-md border border-trace bg-trace/15 px-4 py-3 font-mono text-sm font-bold text-trace transition hover:bg-trace/25 disabled:opacity-50"
        >
          {busy ? "checking..." : "unlock"}
        </button>
      </form>
    </main>
  );
}
