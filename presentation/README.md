# How Far Can a PSRAM-less ESP32 Go? — Presentation

A web-based slide deck (Next.js + framer-motion) for the WiFi CSI HAR paper, by
Muhammad Ahmad (L1F22BSCS0634, Section H9), University of Central Punjab.

## Add your UCP logo
Put your logo file at **`public/ucp.png`**. Until then a text badge is shown automatically.

## Run locally
```bash
npm install
npm run dev      # open http://localhost:3000
```

Presenting: `→` / `Space` next, `←` previous, `F` fullscreen, `Home`/`End` jump.

## Deploy to Vercel
Option A — CLI:
```bash
npm i -g vercel
vercel            # follow prompts (first run links/creates the project)
vercel --prod     # production deployment
```
Option B — GitHub + Vercel dashboard: push this `presentation/` folder to a repo,
"Import Project" on vercel.com, framework auto-detected as Next.js, deploy.

## Add internet images (optional)
Drop image files into `public/web/` and reference them as `/web/<file>` in any slide
under `components/slides/`.
