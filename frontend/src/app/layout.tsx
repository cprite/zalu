import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { PREFS_SCRIPT } from "@/lib/prefs-script";
import { Providers } from "./providers";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: "ZALU · build a camera model from a sentence",
  description: "Describe what to detect, train a small vision model on your footage, and verify it honestly on new real frames.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F7F9FC" },
    { media: "(prefers-color-scheme: dark)", color: "#0E131C" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // data-theme / data-scale / data-motion are written by PREFS_SCRIPT before paint, hence suppressHydrationWarning
    <html lang="en" className={inter.variable} data-theme="light" data-scale="default" data-motion="2" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: PREFS_SCRIPT }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
