#pragma once

#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <chrono>

namespace hybridvectordb {
namespace optimization {

/**
 * @brief SIMD optimization utilities
 */
class SIMDOptimizer {
public:
    /**
     * @brief Check CPU features
     */
    struct CPUFeatures {
        bool has_avx2 = false;
        bool has_fma = false;
        bool has_avx512 = false;
        bool has_sse4_2 = false;
    };
    
    static CPUFeatures detect_cpu_features();
    
    /**
     * @brief Apply best available SIMD optimizations
     */
    static void apply_optimizations(float* data, size_t count, size_t dimension);
    
private:
    static void apply_avx2_optimizations(float* data, size_t count, size_t dimension);
    static void apply_avx512_optimizations(float* data, size_t count, size_t dimension);
    static void apply_sse_optimizations(float* data, size_t count, size_t dimension);
};

/**
 * @brief Memory layout optimization
 */
class MemoryOptimizer {
public:
    /**
     * @brief Optimize memory layout for cache efficiency
     */
    static void optimize_for_cache(float* vectors, size_t count, size_t dimension);
    
    /**
     * @brief Align memory for SIMD operations
     */
    static void align_memory(float* data, size_t count, size_t dimension, size_t alignment = 32);
    
    /**
     * @brief Prefetch memory for better cache performance
     */
    static void prefetch_memory(const float* data, size_t size);
    
    /**
     * @brief Optimize vector storage for sequential access
     */
    static void optimize_sequential_access(float* vectors, size_t count, size_t dimension);
};

/**
 * @brief Parallel processing optimizations
 */
class ParallelOptimizer {
public:
    /**
     * @brief Optimal thread count for current system
     */
    static size_t get_optimal_thread_count();
    
    /**
     * @brief Parallel vector operations
     */
    static void parallel_vector_copy(const float* src, float* dst, size_t count, size_t dimension);
    static void parallel_vector_normalize(float* vectors, size_t count, size_t dimension);
    static void parallel_distance_calculation(const float* query,
                                       const float* vectors,
                                       float* distances,
                                       size_t vector_count,
                                       size_t dimension);
    
    /**
     * @brief Work stealing scheduler for load balancing
     */
    class WorkStealingScheduler {
    private:
        struct WorkItem {
            std::function<void()> task;
            std::atomic<bool> completed{false};
        };
        
        std::vector<WorkItem> work_queue_;
        std::vector<std::thread> workers_;
        std::mutex queue_mutex_;
        std::condition_variable work_available_;
        std::atomic<bool> shutdown_{false};
        size_t num_workers_;
        
    public:
        explicit WorkStealingScheduler(size_t num_workers = 0);
        ~WorkStealingScheduler();
        
        void submit_task(std::function<void()> task);
        void wait_for_completion();
        size_t get_pending_tasks() const;
    };
};

/**
 * @brief Cache optimization
 */
class CacheOptimizer {
public:
    /**
     * @brief Optimize for L1 cache
     */
    static void optimize_for_l1_cache(float* data, size_t count, size_t dimension);
    
    /**
     * @brief Optimize for L2 cache
     */
    static void optimize_for_l2_cache(float* data, size_t count, size_t dimension);
    
    /**
     * @brief Optimize for L3 cache
     */
    static void optimize_for_l3_cache(float* data, size_t count, size_t dimension);
    
    /**
     * @brief Cache-aware blocking for matrix operations
     */
    static void cache_aware_blocking(const float* matrix,
                                 float* result,
                                 size_t rows,
                                 size_t cols,
                                 size_t block_size);
};

/**
 * @brief Performance profiling and benchmarking
 */
class PerformanceProfiler {
private:
    struct ProfileEntry {
        std::string operation;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        size_t data_size;
        std::unordered_map<std::string, std::string> metadata;
    };
    
    std::vector<ProfileEntry> profile_data_;
    std::mutex profile_mutex_;
    
public:
    /**
     * @brief RAII profiler for automatic timing
     */
    class ScopedProfiler {
    private:
        PerformanceProfiler& profiler_;
        ProfileEntry entry_;
        
    public:
        ScopedProfiler(PerformanceProfiler& profiler, 
                     const std::string& operation,
                     size_t data_size = 0,
                     const std::unordered_map<std::string, std::string>& metadata = {});
        ~ScopedProfiler();
    };
    
    /**
     * @brief Start profiling an operation
     */
    void start_profile(const std::string& operation,
                   size_t data_size = 0,
                   const std::unordered_map<std::string, std::string>& metadata = {});
    
    /**
     * @brief End profiling an operation
     */
    void end_profile(const std::string& operation);
    
    /**
     * @brief Get profiling statistics
     */
    std::unordered_map<std::string, double> get_statistics() const;
    
    /**
     * @brief Export profiling data
     */
    void export_data(const std::string& filename);
    
    /**
     * @brief Clear profiling data
     */
    void clear();
};

/**
 * @brief Memory pool for high-performance allocations
 */
class HighPerformanceMemoryPool {
private:
    struct MemoryChunk {
        void* memory;
        size_t size;
        size_t used;
        std::vector<bool> allocation_map;
        std::mutex chunk_mutex;
        
        MemoryChunk(size_t size);
        ~MemoryChunk();
        
        void* allocate(size_t size, size_t alignment);
        void deallocate(void* ptr);
        bool contains(void* ptr) const;
    };
    
    std::vector<std::unique_ptr<MemoryChunk>> chunks_;
    size_t chunk_size_;
    std::mutex pool_mutex_;
    std::atomic<size_t> total_allocated_{0};
    
public:
    explicit HighPerformanceMemoryPool(size_t chunk_size = 64 * 1024 * 1024);
    ~HighPerformanceMemoryPool();
    
    /**
     * @brief Allocate aligned memory
     */
    void* allocate(size_t size, size_t alignment = 32);
    
    /**
     * @brief Deallocate memory
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Get memory usage statistics
     */
    std::unordered_map<std::string, size_t> get_usage_stats() const;
    
    /**
     * @brief Optimize memory layout
     */
    void optimize_layout();
};

/**
 * @brief Zero-copy operations
 */
namespace zero_copy {
    /**
     * @brief Zero-copy vector access
     */
    class ZeroCopyVector {
    private:
        float* data_;
        size_t size_;
        bool owns_data_;
        
    public:
        ZeroCopyVector(float* data, size_t size, bool owns = false)
            : data_(data), size_(size), owns_data_(owns) {}
        
        ~ZeroCopyVector() {
            if (owns_data_ && data_) {
                aligned_free(data_);
            }
        }
        
        float* data() { return data_; }
        const float* data() const { return data_; }
        size_t size() const { return size_; }
        
        // Prevent copying
        ZeroCopyVector(const ZeroCopyVector&) = delete;
        ZeroCopyVector& operator=(const ZeroCopyVector&) = delete;
        
        // Allow moving
        ZeroCopyVector(ZeroCopyVector&& other) noexcept;
        ZeroCopyVector& operator=(ZeroCopyVector&& other) noexcept;
    };
    
    /**
     * @brief Create zero-copy vector
     */
    std::unique_ptr<ZeroCopyVector> create_zero_copy_vector(size_t size);
    
    /**
     * @brief Shared memory operations
     */
    class SharedMemoryManager {
    private:
        std::string shared_name_;
        void* shared_memory_;
        size_t size_;
        bool is_creator_;
        
    public:
        SharedMemoryManager(const std::string& name, size_t size);
        ~SharedMemoryManager();
        
        void* get_memory() const { return shared_memory_; }
        size_t get_size() const { return size_; }
    };
}

/**
 * @brief Aligned memory allocation
 */
void* aligned_malloc(size_t size, size_t alignment);
void aligned_free(void* ptr);

/**
 * @brief CPU instruction set detection
 */
bool has_avx2();
bool has_avx512();
bool has_fma();
bool has_sse4_2();

/**
 * @brief Performance optimization functions
 */
void optimize_batch_size(size_t& optimal_size, 
                     const std::function<double(size_t)>& benchmark_func);
void optimize_thread_count(size_t& optimal_threads);
void optimize_memory_block_size(size_t& optimal_block_size, size_t data_size);

} // namespace optimization
} // namespace hybridvectordb
