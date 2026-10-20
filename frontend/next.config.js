/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep the standard output directory by default. CI/audit builds can set a
  // separate directory to avoid touching existing .next artifacts.
  distDir: process.env.FINRAG_NEXT_DIST_DIR || ".next",
  reactStrictMode: true,
  experimental: {
    // /chat runs several sequential LLM calls and routinely exceeds the
    // default 30s proxy timeout; aborting here surfaces as a 500 in the
    // browser even though the backend eventually returns 200.
    proxyTimeout: 900_000,
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/:path*` },
    ];
  },
};

module.exports = nextConfig;
