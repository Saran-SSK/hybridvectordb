import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// API endpoints for HybridVectorDB
export const hybridVectorDBAPI = {
  // Health check
  health: () => apiClient.get('/health'),
  
  // Metrics
  getMetrics: () => apiClient.get('/metrics'),
  
  // Vector operations
  addVectors: (vectors: number[][], metadata?: any[]) => 
    apiClient.post('/vectors', { vectors, metadata }),
  
  searchVectors: (query: number[], k: number = 10) =>
    apiClient.post('/search', { query, k }),
  
  getVectorCount: () => apiClient.get('/vectors/count'),
  
  // Index operations
  getIndexInfo: () => apiClient.get('/index/info'),
  
  saveIndex: (filepath: string) => apiClient.post('/index/save', { filepath }),
  
  loadIndex: (filepath: string) => apiClient.post('/index/load', { filepath }),
  
  // Benchmark operations
  runBenchmark: (config: any) => apiClient.post('/benchmark', config),
  
  getBenchmarkResults: () => apiClient.get('/benchmark/results'),
  
  // Configuration
  getConfig: () => apiClient.get('/config'),
  
  updateConfig: (config: any) => apiClient.put('/config', config),
  
  // Statistics
  getStats: () => apiClient.get('/stats'),
  
  getPerformanceStats: () => apiClient.get('/stats/performance'),
  
  getMemoryStats: () => apiClient.get('/stats/memory'),
}

// Mock data for development when backend is not available
export const mockAPI = {
  health: () => Promise.resolve({ 
    data: { status: 'healthy', uptime: '14d 7h 23m', version: '0.5.0' } 
  }),
  
  getMetrics: () => Promise.resolve({
    data: {
      totalVectors: 1247832,
      throughput: 2340,
      avgLatency: 1.3,
      memoryUsage: 12.4,
      gpuUsage: 67,
      cpuUsage: 42,
    }
  }),
  
  getStats: () => Promise.resolve({
    data: {
      vector_count: 1247832,
      index_type: 'ivf',
      dimension: 768,
      distance_metric: 'cosine',
      gpu_enabled: true,
      memory_usage: '12.4 GB',
    }
  }),
}

// Export API client that falls back to mock data
export const api = process.env.NODE_ENV === 'development' && !API_BASE_URL.includes('localhost')
  ? mockAPI
  : hybridVectorDBAPI

export default apiClient
