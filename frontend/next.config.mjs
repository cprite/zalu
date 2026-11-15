import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));

/** @type {import('next').NextConfig} */
// devIndicators: false hides the Next.js dev-tools badge; product settings live in the app's own Settings panel.
const nextConfig = { reactStrictMode: true, devIndicators: false, turbopack: { root } };
export default nextConfig;
