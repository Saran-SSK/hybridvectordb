// Mock data for HybridVectorDB Console

// Use stable values to prevent hydration mismatches
const SYSTEM_HEALTH = {
  status: "healthy" as const,
  uptime: "14d 7h 23m",
  version: "0.5.0",
  totalVectors: 1_247_832,
  activeConnections: 24,
  cpuUsage: 42,
  gpuUsage: 67,
  memoryUsed: 12.4, // GB
  memoryTotal: 32.0, // GB
  gpuMemoryUsed: 6.8, // GB
  gpuMemoryTotal: 16.0, // GB
}

const PERFORMANCE_METRICS = [
  { time: "00:00", throughput: 1200, latency: 2.1, cpuOps: 800, gpuOps: 400 },
  { time: "01:00", throughput: 980, latency: 2.4, cpuOps: 620, gpuOps: 360 },
  { time: "02:00", throughput: 750, latency: 2.8, cpuOps: 500, gpuOps: 250 },
  { time: "03:00", throughput: 620, latency: 3.1, cpuOps: 420, gpuOps: 200 },
  { time: "04:00", throughput: 580, latency: 3.3, cpuOps: 380, gpuOps: 200 },
  { time: "05:00", throughput: 650, latency: 3.0, cpuOps: 400, gpuOps: 250 },
  { time: "06:00", throughput: 890, latency: 2.6, cpuOps: 540, gpuOps: 350 },
  { time: "07:00", throughput: 1350, latency: 2.0, cpuOps: 800, gpuOps: 550 },
  { time: "08:00", throughput: 1780, latency: 1.8, cpuOps: 980, gpuOps: 800 },
  { time: "09:00", throughput: 2100, latency: 1.5, cpuOps: 1100, gpuOps: 1000 },
  { time: "10:00", throughput: 2340, latency: 1.3, cpuOps: 1200, gpuOps: 1140 },
  { time: "11:00", throughput: 2500, latency: 1.2, cpuOps: 1250, gpuOps: 1250 },
  { time: "12:00", throughput: 2420, latency: 1.3, cpuOps: 1220, gpuOps: 1200 },
  { time: "13:00", throughput: 2380, latency: 1.4, cpuOps: 1180, gpuOps: 1200 },
  { time: "14:00", throughput: 2200, latency: 1.5, cpuOps: 1100, gpuOps: 1100 },
  { time: "15:00", throughput: 2050, latency: 1.6, cpuOps: 1050, gpuOps: 1000 },
  { time: "16:00", throughput: 1900, latency: 1.7, cpuOps: 980, gpuOps: 920 },
  { time: "17:00", throughput: 1650, latency: 1.9, cpuOps: 850, gpuOps: 800 },
  { time: "18:00", throughput: 1400, latency: 2.1, cpuOps: 750, gpuOps: 650 },
  { time: "19:00", throughput: 1250, latency: 2.3, cpuOps: 700, gpuOps: 550 },
  { time: "20:00", throughput: 1100, latency: 2.4, cpuOps: 650, gpuOps: 450 },
  { time: "21:00", throughput: 1050, latency: 2.5, cpuOps: 620, gpuOps: 430 },
  { time: "22:00", throughput: 980, latency: 2.6, cpuOps: 580, gpuOps: 400 },
  { time: "23:00", throughput: 1100, latency: 2.3, cpuOps: 650, gpuOps: 450 },
]

const BENCHMARK_RESULTS = [
  { dataset: "1K", pythonTime: 0.5, cppTime: 0.2, speedup: 2.5 },
  { dataset: "10K", pythonTime: 2.1, cppTime: 0.6, speedup: 3.5 },
  { dataset: "100K", pythonTime: 18.5, cppTime: 4.8, speedup: 3.9 },
  { dataset: "1M", pythonTime: 32.0, cppTime: 7.2, speedup: 4.4 },
]

const ROUTING_DECISIONS = [
  { time: "Last 1h", cpu: 342, gpu: 518, total: 860 },
  { time: "Last 6h", cpu: 2100, gpu: 3400, total: 5500 },
  { time: "Last 24h", cpu: 8400, gpu: 13200, total: 21600 },
  { time: "Last 7d", cpu: 58000, gpu: 94000, total: 152000 },
]

export type VectorEntry = {
  id: string
  dimensions: number
  indexType: "flat" | "ivf"
  distanceMetric: "l2" | "cosine" | "inner_product"
  metadata: Record<string, string>
  createdAt: string
  lastAccessed: string
}

const VECTORS: VectorEntry[] = [
  {
    id: "vec_001a2b3c",
    dimensions: 768,
    indexType: "ivf",
    distanceMetric: "cosine",
    metadata: { source: "openai", model: "text-embedding-3-small", category: "documents" },
    createdAt: "2026-01-15T10:30:00Z",
    lastAccessed: "2026-02-10T08:15:00Z",
  },
  {
    id: "vec_004d5e6f",
    dimensions: 1536,
    indexType: "flat",
    distanceMetric: "l2",
    metadata: { source: "cohere", model: "embed-english-v3", category: "images" },
    createdAt: "2026-01-20T14:22:00Z",
    lastAccessed: "2026-02-09T22:30:00Z",
  },
  {
    id: "vec_007g8h9i",
    dimensions: 384,
    indexType: "ivf",
    distanceMetric: "inner_product",
    metadata: { source: "sentence-transformers", model: "all-MiniLM-L6-v2", category: "text" },
    createdAt: "2026-01-25T09:10:00Z",
    lastAccessed: "2026-02-10T06:45:00Z",
  },
]

const RECENT_ALERTS = [
  { id: 1, type: "warning" as const, message: "GPU memory usage exceeded 80% threshold", time: "12 min ago" },
  { id: 2, type: "info" as const, message: "Auto-scaling triggered: 3 -> 4 pods", time: "1h ago" },
  { id: 3, type: "success" as const, message: "Benchmark suite completed successfully", time: "2h ago" },
  { id: 4, type: "warning" as const, message: "High latency detected on IVF index rebuild", time: "4h ago" },
  { id: 5, type: "info" as const, message: "New routing strategy deployed: hybrid_v2", time: "6h ago" },
]

const CONFIG_OPTIONS = {
  indexTypes: ["flat", "ivf"] as const,
  distanceMetrics: ["l2", "cosine", "inner_product"] as const,
  routingStrategies: ["performance", "workload", "resource", "hybrid"] as const,
  dataDistributions: ["uniform", "normal", "clustered", "exponential"] as const,
}

// Export stable constants to prevent hydration mismatches
export { 
  SYSTEM_HEALTH as systemHealth,
  PERFORMANCE_METRICS as performanceMetrics,
  BENCHMARK_RESULTS as benchmarkResults,
  ROUTING_DECISIONS as routingDecisions,
  VECTORS as vectors,
  RECENT_ALERTS as recentAlerts,
  CONFIG_OPTIONS as configOptions,
}
