#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace hybridvectordb {

/**
 * @brief Index management for CPU and GPU indexes
 */
class IndexManager {
private:
    struct IndexConfig {
        size_t dimension;
        std::string index_type;
        std::string metric_type;
        bool use_gpu;
    };
    
    IndexConfig config_;
    std::unique_ptr<void> cpu_index_;
    std::unique_ptr<void> gpu_index_;
    std::atomic<size_t> total_vectors_{0};
    mutable std::mutex mutex_;
    
    // Memory tracking
    std::atomic<size_t> memory_usage_bytes_{0};
    
public:
    explicit IndexManager(const struct Config& config);
    ~IndexManager();
    
    void configure(const struct Config& config);
    
    size_t add_vectors(const float* vectors, size_t count, size_t dimension);
    
    const float* get_vectors_zero_copy() const;
    
    size_t get_vector_count() const;
    
    std::unordered_map<std::string, size_t> get_memory_usage() const;
    
    void reset();
    
private:
    void create_cpu_index(const IndexConfig& config);
    void create_gpu_index(const IndexConfig& config);
    void destroy_indexes();
};

} // namespace hybridvectordb
