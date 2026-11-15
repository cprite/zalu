import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const apiUrl = process.env.ZALU_API_URL?.trim();
let apiDestination;
if (apiUrl) {
  const url = new URL(apiUrl);
  if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("ZALU_API_URL must be an HTTP(S) URL without credentials, query, or fragment.");
  }
  apiDestination = url.href.replace(/\/+$/, "");
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  turbopack: { root },
  async rewrites() {
    return apiDestination ? [{ source: "/api/:path*", destination: `${apiDestination}/api/:path*` }] : [];
  },
};
export default nextConfig;
