#include "vector_database.h"
#include "index_manager.h"
#include "memory_manager.h"
#include "search_engine.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>  // SIMD intrinsics

namespace hybridvectordb {

HybridVectorDB::HybridVectorDB(const Config& config) 
    : config_(config) {
    
    // Initialize memory pool (64MB default)
    memory_pool_ = std::make_unique<MemoryPool>(64 * 1024 * 1024);
    
    // Initialize components
    index_manager_ = std::make_unique<IndexManager>(config);
    memory_manager_ = std::make_unique<MemoryManager>(config);
    search_engine_ = std::make_unique<SearchEngine>(config);
    
    // Configure components
    index_manager_->configure(config);
    memory_manager_->configure(config);
    search_engine_->configure(config);
}

HybridVectorDB::~HybridVectorDB() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Cleanup in reverse order
    search_engine_.reset();
    memory_manager_.reset();
    index_manager_.reset();
}

size_t HybridVectorDB::add_vectors(const VectorData* vectors, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate input
        if (!vectors || count == 0) {
            throw HybridVectorDBError("No vectors provided", "INVALID_INPUT");
        }
        
        // Check dimension consistency
        for (size_t i = 0; i < count; ++i) {
            if (vectors[i].embedding.size() != config_.dimension) {
                throw HybridVectorDBError("Vector dimension mismatch", "DIMENSION_MISMATCH");
            }
        }
        
        // Use zero-copy memory allocation
        size_t total_floats = count * config_.dimension;
        float* memory_ptr = memory_pool_->allocate(total_floats);
        
        if (!memory_ptr) {
            throw HybridVectorDBError("Insufficient memory", "MEMORY_ERROR");
        }
        
        // Copy vectors to memory pool (zero-copy)
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(memory_ptr + i * config_.dimension, 
                      vectors[i].embedding.data(), 
                      config_.dimension * sizeof(float));
        }
        
        // Add to index manager
        size_t added = index_manager_->add_vectors(memory_ptr, count, config_.dimension);
        
        // Update memory manager
        memory_manager_->track_allocation(total_floats * sizeof(float));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update metrics
        bool use_gpu = count >= config_.batch_threshold;
        if (use_gpu) {
            metrics_.update_gpu_time(duration.count() / 1000.0);
        } else {
            metrics_.update_cpu_time(duration.count() / 1000.0);
        }
        
        return added;
        
    } catch (const std::exception& e) {
        // Deallocate memory on error
        if (memory_ptr) {
            memory_pool_->deallocate(memory_ptr, total_floats);
        }
        throw HybridVectorDBError(std::string("Add vectors failed: ") + e.what(), "ADD_ERROR");
    }
}

std::vector<SearchResponse> HybridVectorDB::search_vectors(
    const float* query_vectors,
    size_t query_count,
    size_t k,
    bool use_gpu) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<SearchResponse> responses;
    
    try {
        // Validate input
        if (!query_vectors || query_count == 0) {
            throw HybridVectorDBError("No query vectors provided", "INVALID_INPUT");
        }
        
        // Determine execution path
        bool should_use_gpu = use_gpu;
        if (!use_gpu) {
            // Auto-determine based on workload
            should_use_gpu = (query_count >= config_.batch_threshold || 
                             k >= config_.k_threshold);
        }
        
        // Zero-copy query preparation
        size_t total_query_floats = query_count * config_.dimension;
        float* query_memory = memory_pool_->allocate(total_query_floats);
        
        if (!query_memory) {
            throw HybridVectorDBError("Insufficient memory for queries", "MEMORY_ERROR");
        }
        
        // Copy queries to memory pool (zero-copy)
        std::memcpy(query_memory, query_vectors, total_query_floats * sizeof(float));
        
        // Apply SIMD optimizations
        optimization::apply_simd_optimizations(query_memory, query_count, config_.dimension);
        
        // Perform search
        std::vector<SearchResult> results;
        if (should_use_gpu) {
            results = search_engine_->search_gpu(query_memory, query_count, k);
            metrics_.update_gpu_time(0, true);  // Time will be updated below
        } else {
            results = search_engine_->search_cpu(query_memory, query_count, k);
            metrics_.update_cpu_time(0, true);  // Time will be updated below
        }
        
        // Create response objects
        responses.reserve(query_count);
        for (size_t i = 0; i < query_count; ++i) {
            std::vector<SearchResult> query_results;
            query_results.reserve(k);
            
            // Extract results for this query
            size_t start_idx = i * k;
            for (size_t j = 0; j < k && (start_idx + j) < results.size(); ++j) {
                query_results.push_back(results[start_idx + j]);
            }
            
            responses.emplace_back(
                "query_" + std::to_string(i),
                query_results,
                static_cast<size_t>(query_results.size()),
                0.0,  // Will be updated below
                should_use_gpu ? "gpu_optimized" : "cpu_optimized"
            );
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update response times
        double query_time_ms = duration.count() / 1000.0 / query_count;
        for (auto& response : responses) {
            response.search_time_ms = query_time_ms;
        }
        
        // Update metrics
        if (should_use_gpu) {
            metrics_.update_gpu_time(query_time_ms, true);
        } else {
            metrics_.update_cpu_time(query_time_ms, true);
        }
        
        // Cleanup
        memory_pool_->deallocate(query_memory, total_query_floats);
        
        return responses;
        
    } catch (const std::exception& e) {
        // Cleanup on error
        if (query_memory) {
            memory_pool_->deallocate(query_memory, total_query_floats);
        }
        throw HybridVectorDBError(std::string("Search failed: ") + e.what(), "SEARCH_ERROR");
    }
}

PerformanceMetrics HybridVectorDB::get_metrics() const {
    return metrics_;
}

void HybridVectorDB::reset_metrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_ = PerformanceMetrics();
}

std::unordered_map<std::string, std::string> HybridVectorDB::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, std::string> stats;
    stats["total_queries"] = std::to_string(metrics_.total_queries.load());
    stats["cpu_queries"] = std::to_string(metrics_.cpu_queries.load());
    stats["gpu_queries"] = std::to_string(metrics_.gpu_queries.load());
    stats["avg_cpu_time_ms"] = std::to_string(metrics_.cpu_time_ms.load());
    stats["avg_gpu_time_ms"] = std::to_string(metrics_.gpu_time_ms.load());
    stats["speedup"] = std::to_string(metrics_.get_speedup());
    stats["cpu_success_rate"] = std::to_string(metrics_.cpu_success_rate.load());
    stats["gpu_success_rate"] = std::to_string(metrics_.gpu_success_rate.load());
    
    // Add memory usage
    auto memory_usage = get_memory_usage();
    for (const auto& [key, value] : memory_usage) {
        stats[key] = std::to_string(value);
    }
    
    return stats;
}

void HybridVectorDB::configure(const Config& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    
    // Reconfigure components
    index_manager_->configure(config);
    memory_manager_->configure(config);
    search_engine_->configure(config);
}

std::unordered_map<std::string, double> HybridVectorDB::optimize_performance(const std::string& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, double> results;
    
    if (operation == "batch_size") {
        // Test different batch sizes
        std::vector<size_t> test_sizes = {16, 32, 64, 128, 256, 512};
        std::vector<double> times;
        
        for (size_t batch_size : test_sizes) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate search operation
            std::vector<float> queries(batch_size * config_.dimension, 0.0f);
            auto search_results = search_vectors(queries.data(), batch_size, 10, false);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        // Find optimal batch size
        auto min_it = std::min_element(times.begin(), times.end());
        size_t optimal_idx = std::distance(times.begin(), min_it);
        size_t optimal_batch_size = test_sizes[optimal_idx];
        
        results["optimal_batch_size"] = static_cast<double>(optimal_batch_size);
        results["optimal_time_ms"] = *min_it;
        results["throughput_vps"] = (optimal_batch_size * 1000.0) / *min_it;
        
    } else if (operation == "memory_layout") {
        // Memory layout optimization
        size_t vector_count = index_manager_->get_vector_count();
        if (vector_count > 0) {
            float* vectors = get_vectors_zero_copy(vector_count);
            optimization::optimize_memory_layout(vectors, vector_count, config_.dimension);
        }
        
        results["memory_optimized"] = true;
        results["vector_count"] = static_cast<double>(vector_count);
        
    } else {
        results["error"] = std::string("Unknown operation: ") + operation;
    }
    
    return results;
}

const float* HybridVectorDB::get_vectors_zero_copy(size_t& count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    count = index_manager_->get_vector_count();
    if (count == 0) {
        return nullptr;
    }
    
    // Allocate from memory pool for zero-copy access
    size_t total_floats = count * config_.dimension;
    float* memory_ptr = memory_pool_->allocate(total_floats);
    
    if (!memory_ptr) {
        throw HybridVectorDBError("Insufficient memory for zero-copy access", "MEMORY_ERROR");
    }
    
    // Get vectors from index manager
    const float* vectors = index_manager_->get_vectors_zero_copy();
    
    // Copy to memory pool
    std::memcpy(memory_ptr, vectors, total_floats * sizeof(float));
    
    return memory_ptr;
}

std::unordered_map<std::string, size_t> HybridVectorDB::get_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, size_t> usage;
    
    // Memory pool usage
    if (memory_pool_) {
        usage["memory_pool_capacity"] = memory_pool_->capacity;
        usage["memory_pool_used"] = memory_pool_->used;
        usage["memory_pool_free"] = memory_pool_->capacity - memory_pool_->used;
    }
    
    // Index manager usage
    auto index_usage = index_manager_->get_memory_usage();
    usage.insert(index_usage.begin(), index_usage.end());
    
    // Memory manager usage
    auto memory_manager_usage = memory_manager_->get_memory_usage();
    usage.insert(memory_manager_usage.begin(), memory_manager_usage.end());
    
    return usage;
}

std::unordered_map<std::string, double> HybridVectorDB::benchmark(
    size_t num_vectors, size_t num_queries, size_t k) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, double> results;
    
    // Prepare test data
    std::vector<VectorData> test_vectors;
    test_vectors.reserve(num_vectors);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> embedding(config_.dimension);
        for (size_t j = 0; j < config_.dimension; ++j) {
            embedding[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        test_vectors.emplace_back("vec_" + std::to_string(i), embedding);
    }
    
    // Add vectors
    auto start_add = std::chrono::high_resolution_clock::now();
    add_vectors(test_vectors.data(), test_vectors.size());
    auto end_add = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::microseconds>(end_add - start_add);
    
    // Benchmark searches
    std::vector<double> cpu_times, gpu_times;
    
    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<float> query(config_.dimension);
        for (size_t j = 0; j < config_.dimension; ++j) {
            query[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        // CPU benchmark
        auto start_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_results = search_vectors(query.data(), 1, k, false);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
        cpu_times.push_back(cpu_time.count() / 1000.0);
        
        // GPU benchmark (if available)
        if (config_.use_gpu) {
            auto start_gpu = std::chrono::high_resolution_clock::now();
            auto gpu_results = search_vectors(query.data(), 1, k, true);
            auto end_gpu = std::chrono::high_resolution_clock::now();
            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
            gpu_times.push_back(gpu_time.count() / 1000.0);
        }
    }
    
    // Calculate statistics
    double avg_cpu_time = 0.0, avg_gpu_time = 0.0;
    for (double time : cpu_times) {
        avg_cpu_time += time;
    }
    for (double time : gpu_times) {
        avg_gpu_time += time;
    }
    
    if (!cpu_times.empty()) {
        avg_cpu_time /= cpu_times.size();
    }
    if (!gpu_times.empty()) {
        avg_gpu_time /= gpu_times.size();
    }
    
    results["num_vectors"] = static_cast<double>(num_vectors);
    results["num_queries"] = static_cast<double>(num_queries);
    results["k"] = static_cast<double>(k);
    results["add_time_ms"] = static_cast<double>(add_time.count() / 1000.0);
    results["avg_cpu_time_ms"] = avg_cpu_time;
    results["avg_gpu_time_ms"] = avg_gpu_time;
    results["speedup"] = avg_cpu_time > 0 ? avg_cpu_time / avg_gpu_time : 1.0;
    results["throughput_qps_cpu"] = (num_queries * 1000.0) / avg_cpu_time;
    results["throughput_qps_gpu"] = avg_gpu_time > 0 ? (num_queries * 1000.0) / avg_gpu_time : 0.0;
    
    return results;
}

// Factory function
std::unique_ptr<HybridVectorDB> create_vector_database(const Config& config) {
    return std::make_unique<HybridVectorDB>(config);
}

} // namespace hybridvectordb
