#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace hybridvectordb {

/**
 * @brief High-performance search engine with SIMD optimizations
 */
class SearchEngine {
private:
    struct SearchConfig {
        size_t dimension;
        std::string metric_type;
        bool use_gpu;
        size_t batch_threshold;
        size_t k_threshold;
    };
    
    SearchConfig config_;
    std::unique_ptr<void> cpu_searcher_;
    std::unique_ptr<void> gpu_searcher_;
    std::atomic<uint64_t> search_count_{0};
    mutable std::mutex mutex_;
    
    // Performance optimization
    struct SearchCache {
        struct CacheEntry {
            std::vector<float> query;
            std::vector<SearchResult> results;
            size_t k;
            std::chrono::steady_clock::time_point timestamp;
            
            CacheEntry(const std::vector<float>& q, size_t k_val)
                : query(q), k(k_val), timestamp(std::chrono::steady_clock::now()) {}
        };
        
        std::vector<CacheEntry> entries;
        size_t max_size;
        std::chrono::seconds ttl;
        
        SearchCache(size_t size = 1000, std::chrono::seconds ttl = std::chrono::seconds(300))
            : max_size(size), ttl(ttl) {}
    };
    
    std::unique_ptr<SearchCache> search_cache_;
    
public:
    explicit SearchEngine(const struct Config& config);
    ~SearchEngine();
    
    void configure(const struct Config& config);
    
    /**
     * @brief Search using CPU with SIMD optimizations
     */
    std::vector<SearchResult> search_cpu(const float* query_vectors, 
                                     size_t query_count, 
                                     size_t k);
    
    /**
     * @brief Search using GPU (if available)
     */
    std::vector<SearchResult> search_gpu(const float* query_vectors, 
                                     size_t query_count, 
                                     size_t k);
    
    /**
     * @brief Hybrid search with automatic CPU/GPU selection
     */
    std::vector<SearchResult> search_hybrid(const float* query_vectors,
                                        size_t query_count,
                                        size_t k);
    
    /**
     * @brief Get search statistics
     */
    std::unordered_map<std::string, uint64_t> get_stats() const;
    
    /**
     * @brief Clear search cache
     */
    void clear_cache();
    
    /**
     * @brief Optimize search performance
     */
    void optimize_search_performance();
    
private:
    void create_cpu_searcher(const SearchConfig& config);
    void create_gpu_searcher(const SearchConfig& config);
    void destroy_searchers();
    
    // SIMD-optimized distance calculations
    std::vector<SearchResult> search_simd_l2(const float* query,
                                              const float* vectors,
                                              size_t vector_count,
                                              size_t dimension,
                                              size_t k);
    
    std::vector<SearchResult> search_simd_inner_product(const float* query,
                                                      const float* vectors,
                                                      size_t vector_count,
                                                      size_t dimension,
                                                      size_t k);
    
    std::vector<SearchResult> search_simd_cosine(const float* query,
                                               const float* vectors,
                                               size_t vector_count,
                                               size_t dimension,
                                               size_t k);
    
    // Cache operations
    std::vector<SearchResult> check_cache(const std::vector<float>& query, size_t k);
    void add_to_cache(const std::vector<float>& query, 
                    const std::vector<SearchResult>& results, 
                    size_t k);
    void cleanup_cache();
};

/**
 * @brief SIMD-optimized distance calculations
 */
namespace distance {
    /**
     * @brief L2 distance with AVX2
     */
    float l2_distance_avx2(const float* a, const float* b, size_t count);
    
    /**
     * @brief Inner product with AVX2
     */
    float inner_product_avx2(const float* a, const float* b, size_t count);
    
    /**
     * @brief Cosine similarity with AVX2
     */
    float cosine_similarity_avx2(const float* a, const float* b, size_t count);
    
    /**
     * @brief Batch distance calculations
     */
    void batch_l2_distances(const float* query,
                          const float* vectors,
                          float* distances,
                          size_t vector_count,
                          size_t dimension);
    
    void batch_inner_products(const float* query,
                           const float* vectors,
                           float* products,
                           size_t vector_count,
                           size_t dimension);
}

/**
 * @brief Search result ranking and filtering
 */
namespace ranking {
    /**
     * @brief Partial sort for top-k selection
     */
    void partial_sort(float* distances, size_t* indices, size_t n, size_t k);
    
    /**
     * @brief Heap-based top-k selection
     */
    void heap_select_top_k(const float* distances,
                        size_t* indices,
                        size_t n,
                        size_t k);
    
    /**
     * @brief Quickselect for top-k
     */
    void quickselect_top_k(float* distances,
                        size_t* indices,
                        size_t n,
                        size_t k);
}

} // namespace hybridvectordb
