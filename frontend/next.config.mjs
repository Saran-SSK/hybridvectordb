/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/api/:path*', // Proxy to HybridVectorDB backend
      },
      {
        source: '/health',
        destination: 'http://localhost:8080/health',
      },
      {
        source: '/metrics',
        destination: 'http://localhost:8080/metrics',
      },
    ]
  },
  env: {
    HYBRIDVECTORDB_API_URL: process.env.HYBRIDVECTORDB_API_URL || 'http://localhost:8080',
    HYBRIDVECTORDB_WS_URL: process.env.HYBRIDVECTORDB_WS_URL || 'ws://localhost:8080',
  }
}

export default nextConfig
