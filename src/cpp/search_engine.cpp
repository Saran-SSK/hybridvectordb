#include "search_engine.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace hybridvectordb {

SearchEngine::SearchEngine(const Config& config) : config_(config) {
    initialize_search_cache();
}

SearchEngine::~SearchEngine() {
    cleanup_search_cache();
}

void SearchEngine::initialize_search_cache() {
    cache_size_ = 1000; // Default cache size
    search_cache_.reserve(cache_size_);
    std::cout << "Search cache initialized with size: " << cache_size_ << std::endl;
}

void SearchEngine::cleanup_search_cache() {
    search_cache_.clear();
}

std::vector<SearchResult> SearchEngine::search_cpu(const float* query, size_t query_count, 
                                                   size_t k, faiss::Index* index) {
    std::vector<SearchResult> results;
    
    if (!query || query_count == 0 || k == 0 || !index) {
        return results;
    }
    
    try {
        // Allocate buffers for results
        std::vector<float> distances(query_count * k);
        std::vector<int64_t> labels(query_count * k);
        
        // Perform search
        index->search(query_count, query, k, distances.data(), labels.data());
        
        // Convert to SearchResult format
        for (size_t q = 0; q < query_count; ++q) {
            for (size_t i = 0; i < k; ++i) {
                size_t idx = q * k + i;
                if (labels[idx] >= 0) { // Valid result
                    SearchResult result;
                    result.id = std::to_string(labels[idx]);
                    result.distance = distances[idx];
                    results.push_back(result);
                }
            }
        }
        
    } catch (const std::exception& e) {
        throw HybridVectorDBError("CPU search failed: " + std::string(e.what()), "CPU_SEARCH_ERROR");
    }
    
    return results;
}

std::vector<SearchResult> SearchEngine::search_gpu(const float* query, size_t query_count, 
                                                   size_t k, faiss::Index* index) {
    std::vector<SearchResult> results;
    
    if (!query || query_count == 0 || k == 0 || !index) {
        return results;
    }
    
    try {
        // Allocate buffers for results
        std::vector<float> distances(query_count * k);
        std::vector<int64_t> labels(query_count * k);
        
        // Perform GPU search
        index->search(query_count, query, k, distances.data(), labels.data());
        
        // Convert to SearchResult format
        for (size_t q = 0; q < query_count; ++q) {
            for (size_t i = 0; i < k; ++i) {
                size_t idx = q * k + i;
                if (labels[idx] >= 0) { // Valid result
                    SearchResult result;
                    result.id = std::to_string(labels[idx]);
                    result.distance = distances[idx];
                    results.push_back(result);
                }
            }
        }
        
    } catch (const std::exception& e) {
        throw HybridVectorDBError("GPU search failed: " + std::string(e.what()), "GPU_SEARCH_ERROR");
    }
    
    return results;
}

std::vector<SearchResult> SearchEngine::search_hybrid(const float* query, size_t query_count, 
                                                     size_t k, faiss::Index* cpu_index, 
                                                     faiss::Index* gpu_index) {
    // Decide whether to use CPU or GPU based on query characteristics
    bool use_gpu = should_use_gpu(query_count, k);
    
    if (use_gpu && gpu_index) {
        return search_gpu(query, query_count, k, gpu_index);
    } else if (cpu_index) {
        return search_cpu(query, query_count, k, cpu_index);
    } else {
        throw HybridVectorDBError("No suitable index available for search", "NO_INDEX_ERROR");
    }
}

bool SearchEngine::should_use_gpu(size_t query_count, size_t k) {
    // Simple heuristic: use GPU for larger queries or higher k values
    return (query_count >= 10) || (k >= 50);
}

std::vector<SearchResult> SearchEngine::search_with_cache(const float* query, size_t query_count, 
                                                         size_t k, faiss::Index* index) {
    // Create cache key
    std::string cache_key = create_cache_key(query, query_count, k);
    
    // Check cache
    auto it = search_cache_.find(cache_key);
    if (it != search_cache_.end()) {
        return it->second;
    }
    
    // Perform search
    std::vector<SearchResult> results = search_cpu(query, query_count, k, index);
    
    // Cache results
    cache_results(cache_key, results);
    
    return results;
}

std::string SearchEngine::create_cache_key(const float* query, size_t query_count, size_t k) {
    // Simple hash-based cache key
    std::string key = std::to_string(query_count) + "_" + std::to_string(k);
    
    // Add first few elements of query for better uniqueness
    size_t hash_elements = std::min(query_count * k, static_cast<size_t>(10));
    for (size_t i = 0; i < hash_elements; ++i) {
        key += "_" + std::to_string(static_cast<int>(query[i] * 1000));
    }
    
    return key;
}

void SearchEngine::cache_results(const std::string& key, const std::vector<SearchResult>& results) {
    // Simple LRU cache: if cache is full, remove oldest entry
    if (search_cache_.size() >= cache_size_) {
        search_cache_.erase(search_cache_.begin());
    }
    
    search_cache_[key] = results;
}

void SearchEngine::clear_cache() {
    search_cache_.clear();
}

size_t SearchEngine::get_cache_size() const {
    return search_cache_.size();
}

// distance namespace implementation
namespace distance {

float compute_l2_distance(const float* a, const float* b, size_t dimension) {
    float sum = 0.0f;
    
    for (size_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

float compute_inner_product(const float* a, const float* b, size_t dimension) {
    float sum = 0.0f;
    
    for (size_t i = 0; i < dimension; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

float compute_cosine_similarity(const float* a, const float* b, size_t dimension) {
    float dot_product = compute_inner_product(a, b, dimension);
    
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < dimension; ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

float compute_distance(const float* a, const float* b, size_t dimension, const std::string& metric) {
    if (metric == "l2") {
        return compute_l2_distance(a, b, dimension);
    } else if (metric == "inner_product") {
        return -compute_inner_product(a, b, dimension); // Negative for similarity
    } else if (metric == "cosine") {
        return 1.0f - compute_cosine_similarity(a, b, dimension); // Convert to distance
    } else {
        throw HybridVectorDBError("Unsupported distance metric: " + metric, "UNSUPPORTED_METRIC");
    }
}

} // namespace distance

// ranking namespace implementation
namespace ranking {

void sort_by_distance(std::vector<SearchResult>& results, bool ascending) {
    if (ascending) {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.distance < b.distance;
            });
    } else {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.distance > b.distance;
            });
    }
}

void filter_by_threshold(std::vector<SearchResult>& results, float threshold, bool max_distance) {
    if (max_distance) {
        results.erase(
            std::remove_if(results.begin(), results.end(),
                [threshold](const SearchResult& r) { return r.distance > threshold; }),
            results.end()
        );
    } else {
        results.erase(
            std::remove_if(results.begin(), results.end(),
                [threshold](const SearchResult& r) { return r.distance < threshold; }),
            results.end()
        );
    }
}

std::vector<SearchResult> get_top_k(std::vector<SearchResult>& results, size_t k) {
    if (k >= results.size()) {
        return results;
    }
    
    // Sort by distance (ascending for similarity search)
    sort_by_distance(results, true);
    
    // Return top k results
    return std::vector<SearchResult>(results.begin(), results.begin() + k);
}

void rerank_by_metadata(std::vector<SearchResult>& results, 
                        const std::function<float(const SearchResult&)>& score_func) {
    // Apply custom scoring function
    for (auto& result : results) {
        result.distance = score_func(result);
    }
    
    // Re-sort based on new scores
    sort_by_distance(results, true);
}

} // namespace ranking

} // namespace hybridvectordb
