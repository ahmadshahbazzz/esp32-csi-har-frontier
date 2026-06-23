import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "How Far Can a PSRAM-less ESP32 Go? | WiFi CSI HAR",
  description:
    "A TinyML deployment study for WiFi CSI human activity recognition on the bare classic ESP32. Muhammad Ahmad, University of Central Punjab.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
