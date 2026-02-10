#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

// Forward declarations
class IndexManager;
class MemoryManager;
class SearchEngine;

namespace hybridvectordb {

/**
 * @brief Configuration for vector database
 */
struct Config {
    size_t dimension = 128;
    std::string index_type = "flat";
    std::string metric_type = "l2";
    bool use_gpu = false;
    size_t batch_threshold = 32;
    size_t k_threshold = 50;
    size_t dataset_threshold = 100000;
    double gpu_util_limit = 0.8;
};

/**
 * @brief Vector data with metadata
 */
struct VectorData {
    std::string id;
    std::vector<float> embedding;
    std::unordered_map<std::string, std::string> metadata;
    
    VectorData(const std::string& id, const std::vector<float>& embedding)
        : id(id), embedding(embedding) {}
};

/**
 * @brief Search result
 */
struct SearchResult {
    std::string id;
    float distance;
    std::unordered_map<std::string, std::string> metadata;
    
    SearchResult(const std::string& id, float distance)
        : id(id), distance(distance) {}
};

/**
 * @brief Search response
 */
struct SearchResponse {
    std::string query_id;
    std::vector<SearchResult> results;
    size_t total_results;
    double search_time_ms;
    std::string index_used;
    
    SearchResponse(const std::string& query_id, 
                const std::vector<SearchResult>& results,
                double search_time_ms,
                const std::string& index_used)
        : query_id(query_id), results(results), total_results(results.size()),
          search_time_ms(search_time_ms), index_used(index_used) {}
};

/**
 * @brief Performance metrics
 */
struct PerformanceMetrics {
    std::atomic<uint64_t> total_queries{0};
    std::atomic<uint64_t> cpu_queries{0};
    std::atomic<uint64_t> gpu_queries{0};
    std::atomic<double> total_search_time_ms{0.0};
    std::atomic<double> cpu_time_ms{0.0};
    std::atomic<double> gpu_time_ms{0.0};
    std::atomic<double> cpu_success_rate{1.0};
    std::atomic<double> gpu_success_rate{1.0};
    
    void update_cpu_time(double time_ms, bool success = true) {
        cpu_time_ms.store((cpu_time_ms.load() * cpu_queries.load() + time_ms) / (cpu_queries.load() + 1));
        cpu_queries.fetch_add(1);
        total_queries.fetch_add(1);
        if (!success) {
            cpu_success_rate.store((cpu_success_rate.load() * (cpu_queries.load() - 1)) / cpu_queries.load());
        }
    }
    
    void update_gpu_time(double time_ms, bool success = true) {
        gpu_time_ms.store((gpu_time_ms.load() * gpu_queries.load() + time_ms) / (gpu_queries.load() + 1));
        gpu_queries.fetch_add(1);
        total_queries.fetch_add(1);
        if (!success) {
            gpu_success_rate.store((gpu_success_rate.load() * (gpu_queries.load() - 1)) / gpu_queries.load());
        }
    }
    
    double get_speedup() const {
        double cpu_time = cpu_time_ms.load();
        double gpu_time = gpu_time_ms.load();
        return cpu_time > 0 ? cpu_time / gpu_time : 1.0;
    }
};

/**
 * @brief Main HybridVectorDB C++ class
 */
class HybridVectorDB {
private:
    Config config_;
    std::unique_ptr<IndexManager> index_manager_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<SearchEngine> search_engine_;
    PerformanceMetrics metrics_;
    mutable std::mutex mutex_;
    
    // Zero-copy memory pool
    struct MemoryPool {
        std::vector<float> data;
        size_t capacity;
        size_t used;
        std::mutex pool_mutex;
        
        MemoryPool(size_t capacity) : capacity(capacity), used(0) {
            data.reserve(capacity);
        }
        
        float* allocate(size_t size) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (used + size > capacity) {
                return nullptr;
            }
            float* ptr = data.data() + used;
            used += size;
            return ptr;
        }
        
        void deallocate(float* ptr, size_t size) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (ptr >= data.data() && ptr < data.data() + capacity) {
                used -= size;
            }
        }
    };
    
    std::unique_ptr<MemoryPool> memory_pool_;
    
public:
    /**
     * @brief Constructor
     */
    explicit HybridVectorDB(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~HybridVectorDB();
    
    /**
     * @brief Add vectors to the database
     * @param vectors Vector data to add
     * @param count Number of vectors
     * @return Number of vectors successfully added
     */
    size_t add_vectors(const VectorData* vectors, size_t count);
    
    /**
     * @brief Search for similar vectors
     * @param query_vectors Query vectors
     * @param query_count Number of query vectors
     * @param k Number of results to return
     * @param use_gpu Force GPU usage
     * @return Search responses
     */
    std::vector<SearchResponse> search_vectors(
        const float* query_vectors,
        size_t query_count,
        size_t k,
        bool use_gpu = false
    );
    
    /**
     * @brief Get performance metrics
     */
    PerformanceMetrics get_metrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void reset_metrics();
    
    /**
     * @brief Get database statistics
     */
    std::unordered_map<std::string, std::string> get_stats() const;
    
    /**
     * @brief Configure the database
     */
    void configure(const Config& config);
    
    /**
     * @brief Optimize performance
     */
    std::unordered_map<std::string, double> optimize_performance(const std::string& operation);
    
    /**
     * @brief Zero-copy vector access
     */
    const float* get_vectors_zero_copy(size_t& count) const;
    
    /**
     * @brief Memory usage information
     */
    std::unordered_map<std::string, size_t> get_memory_usage() const;
    
    /**
     * @brief Benchmark performance
     */
    std::unordered_map<std::string, double> benchmark(size_t num_vectors, 
                                                     size_t num_queries,
                                                     size_t k);
};

/**
 * @brief Factory function to create HybridVectorDB instance
 */
std::unique_ptr<HybridVectorDB> create_vector_database(const Config& config);

/**
 * @brief Performance optimization utilities
 */
namespace optimization {
    /**
     * @brief Optimize batch size for given operation
     */
    size_t optimize_batch_size(const std::function<double(size_t)>& operation,
                           const std::vector<size_t>& test_sizes);
    
    /**
     * @brief Optimize memory layout
     */
    void optimize_memory_layout(float* data, size_t count, size_t dimension);
    
    /**
     * @brief SIMD optimizations
     */
    void apply_simd_optimizations(float* data, size_t count, size_t dimension);
}

/**
 * @brief Error handling
 */
class HybridVectorDBError : public std::exception {
private:
    std::string message_;
    std::string error_code_;
    
public:
    HybridVectorDBError(const std::string& message, 
                      const std::string& error_code = "")
        : message_(message), error_code_(error_code) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    const std::string& error_code() const {
        return error_code_;
    }
};

} // namespace hybridvectordb
