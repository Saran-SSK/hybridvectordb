#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace hybridvectordb {

/**
 * @brief Memory management for zero-copy operations
 */
class MemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_access;
        
        MemoryBlock(void* p, size_t s) 
            : ptr(p), size(s), in_use(false) {
            last_access = std::chrono::steady_clock::now();
        }
    };
    
    std::vector<MemoryBlock> memory_blocks_;
    std::unordered_map<void*, size_t> block_sizes_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    mutable std::mutex mutex_;
    
    // GPU memory tracking (if available)
    bool gpu_available_;
    size_t gpu_total_memory_{0};
    std::atomic<size_t> gpu_used_memory_{0};
    
public:
    explicit MemoryManager(const struct Config& config);
    ~MemoryManager();
    
    /**
     * @brief Allocate aligned memory for SIMD operations
     */
    void* allocate_aligned(size_t size, size_t alignment = 32);
    
    /**
     * @brief Allocate memory from pool
     */
    void* allocate(size_t size);
    
    /**
     * @brief Deallocate memory
     */
    void deallocate(void* ptr, size_t size);
    
    /**
     * @brief Get memory usage statistics
     */
    std::unordered_map<std::string, size_t> get_memory_usage() const;
    
    /**
     * @brief Optimize memory layout
     */
    void optimize_memory_layout();
    
    /**
     * @brief Get GPU memory information
     */
    std::unordered_map<std::string, size_t> get_gpu_memory_info() const;
    
    /**
     * @brief Reset memory manager
     */
    void reset();
    
    /**
     * @brief Check if pointer is from this manager
     */
    bool is_managed_ptr(void* ptr) const;
    
private:
    void* allocate_from_system(size_t size);
    void deallocate_to_system(void* ptr, size_t size);
    MemoryBlock* find_block(void* ptr);
    void merge_free_blocks();
};

/**
 * @brief SIMD-optimized memory operations
 */
namespace memory_ops {
    /**
     * @brief Copy vector with SIMD optimizations
     */
    void copy_vector_simd(const float* src, float* dst, size_t count);
    
    /**
     * @brief Compute dot product with SIMD
     */
    float dot_product_simd(const float* a, const float* b, size_t count);
    
    /**
     * @brief Compute L2 distance with SIMD
     */
    float l2_distance_simd(const float* a, const float* b, size_t count);
    
    /**
     * @brief Batch vector operations
     */
    void batch_copy_vectors(const float* src, float* dst, size_t count, size_t dimension);
    void batch_normalize_vectors(float* vectors, size_t count, size_t dimension);
}

} // namespace hybridvectordb
