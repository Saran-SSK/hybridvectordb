#include "optimizations.h"
#include <cpuid.h>
#include <thread>
#include <cstring>
#include <cstdlib>

namespace hybridvectordb {
namespace optimization {

// SIMD Optimizer Implementation
SIMDOptimizer::CPUFeatures SIMDOptimizer::detect_cpu_features() {
    CPUFeatures features;
    
    // Check CPUID for instruction sets
    int cpu_info[4];
    __cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    
    if (n_ids >= 1) {
        __cpuid(cpu_info, 1);
        
        // Check for SSE4.2
        if (cpu_info[2] & (1 << 20)) {
            features.has_sse4_2 = true;
        }
        
        // Check for AVX
        if (cpu_info[2] & (1 << 28)) {
            __cpuid(cpu_info, 7);
            
            // Check for AVX2
            if (cpu_info[1] & (1 << 5)) {
                features.has_avx2 = true;
            }
            
            // Check for FMA
            if (cpu_info[1] & (1 << 12)) {
                features.has_fma = true;
            }
            
            // Check for AVX512
            if (cpu_info[1] & (1 << 16)) {
                features.has_avx512 = true;
            }
        }
    }
    
    return features;
}

void SIMDOptimizer::apply_optimizations(float* data, size_t count, size_t dimension) {
    CPUFeatures features = detect_cpu_features();
    
    if (features.has_avx512) {
        apply_avx512_optimizations(data, count, dimension);
    } else if (features.has_avx2) {
        apply_avx2_optimizations(data, count, dimension);
    } else if (features.has_sse4_2) {
        apply_sse_optimizations(data, count, dimension);
    }
}

void SIMDOptimizer::apply_avx2_optimizations(float* data, size_t count, size_t dimension) {
    // Align data for AVX2 operations (32-byte alignment)
    for (size_t i = 0; i < count; ++i) {
        float* vector = data + i * dimension;
        
        // Normalize vectors for better numerical stability
        float sum_squares = 0.0f;
        size_t avx_end = dimension - (dimension % 8);
        
        // Use AVX2 for bulk of the computation
        __m256 sum_vec = _mm256_setzero_ps();
        for (size_t j = 0; j < avx_end; j += 8) {
            __m256 vec = _mm256_loadu_ps(vector + j);
            sum_vec = _mm256_fmadd_ps(vec, vec, sum_vec);
        }
        
        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        for (int k = 0; k < 8; ++k) {
            sum_squares += sum_array[k];
        }
        
        // Handle remaining elements
        for (size_t j = avx_end; j < dimension; ++j) {
            sum_squares += vector[j] * vector[j];
        }
        
        // Normalize
        float norm = std::sqrt(sum_squares);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            
            for (size_t j = 0; j < avx_end; j += 8) {
                __m256 vec = _mm256_loadu_ps(vector + j);
                __m256 inv_norm_vec = _mm256_set1_ps(inv_norm);
                __m256 normalized = _mm256_mul_ps(vec, inv_norm_vec);
                _mm256_storeu_ps(vector + j, normalized);
            }
            
            for (size_t j = avx_end; j < dimension; ++j) {
                vector[j] *= inv_norm;
            }
        }
    }
}

void SIMDOptimizer::apply_avx512_optimizations(float* data, size_t count, size_t dimension) {
    // Similar to AVX2 but with 512-bit vectors
    for (size_t i = 0; i < count; ++i) {
        float* vector = data + i * dimension;
        
        float sum_squares = 0.0f;
        size_t avx512_end = dimension - (dimension % 16);
        
        __m512 sum_vec = _mm512_setzero_ps();
        for (size_t j = 0; j < avx512_end; j += 16) {
            __m512 vec = _mm512_loadu_ps(vector + j);
            sum_vec = _mm512_fmadd_ps(vec, vec, sum_vec);
        }
        
        // Horizontal sum for AVX512
        float sum_array[16];
        _mm512_storeu_ps(sum_array, sum_vec);
        for (int k = 0; k < 16; ++k) {
            sum_squares += sum_array[k];
        }
        
        // Handle remaining elements
        for (size_t j = avx512_end; j < dimension; ++j) {
            sum_squares += vector[j] * vector[j];
        }
        
        // Normalize
        float norm = std::sqrt(sum_squares);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            
            for (size_t j = 0; j < avx512_end; j += 16) {
                __m512 vec = _mm512_loadu_ps(vector + j);
                __m512 inv_norm_vec = _mm512_set1_ps(inv_norm);
                __m512 normalized = _mm512_mul_ps(vec, inv_norm_vec);
                _mm512_storeu_ps(vector + j, normalized);
            }
            
            for (size_t j = avx512_end; j < dimension; ++j) {
                vector[j] *= inv_norm;
            }
        }
    }
}

void SIMDOptimizer::apply_sse_optimizations(float* data, size_t count, size_t dimension) {
    // Fallback to SSE4.2 optimizations
    for (size_t i = 0; i < count; ++i) {
        float* vector = data + i * dimension;
        
        float sum_squares = 0.0f;
        size_t sse_end = dimension - (dimension % 4);
        
        __m128 sum_vec = _mm_setzero_ps();
        for (size_t j = 0; j < sse_end; j += 4) {
            __m128 vec = _mm_loadu_ps(vector + j);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(vec, vec));
        }
        
        // Horizontal sum
        float sum_array[4];
        _mm_storeu_ps(sum_array, sum_vec);
        for (int k = 0; k < 4; ++k) {
            sum_squares += sum_array[k];
        }
        
        // Handle remaining elements
        for (size_t j = sse_end; j < dimension; ++j) {
            sum_squares += vector[j] * vector[j];
        }
        
        // Normalize
        float norm = std::sqrt(sum_squares);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            
            for (size_t j = 0; j < sse_end; j += 4) {
                __m128 vec = _mm_loadu_ps(vector + j);
                __m128 inv_norm_vec = _mm_set1_ps(inv_norm);
                __m128 normalized = _mm_mul_ps(vec, inv_norm_vec);
                _mm_storeu_ps(vector + j, normalized);
            }
            
            for (size_t j = sse_end; j < dimension; ++j) {
                vector[j] *= inv_norm;
            }
        }
    }
}

// Memory Optimizer Implementation
void MemoryOptimizer::optimize_for_cache(float* vectors, size_t count, size_t dimension) {
    // Reorganize data for better cache line utilization
    size_t cache_line_size = 64; // Typical cache line size
    size_t vector_size_bytes = dimension * sizeof(float);
    
    // Pad vectors to cache line boundaries
    size_t padded_size = ((vector_size_bytes + cache_line_size - 1) / cache_line_size) * cache_line_size;
    
    std::vector<float> optimized_data(count * padded_size / sizeof(float));
    
    for (size_t i = 0; i < count; ++i) {
        float* src = vectors + i * dimension;
        float* dst = optimized_data.data() + i * (padded_size / sizeof(float));
        
        // Copy vector data
        std::memcpy(dst, src, vector_size_bytes);
        
        // Zero out padding
        std::memset(dst + dimension, 0, padded_size - vector_size_bytes);
    }
    
    // Copy back to original location
    std::memcpy(vectors, optimized_data.data(), count * vector_size_bytes);
}

void MemoryOptimizer::align_memory(float* data, size_t count, size_t dimension, size_t alignment) {
    size_t total_size = count * dimension * sizeof(float);
    
    // Check if already aligned
    if (reinterpret_cast<uintptr_t>(data) % alignment == 0) {
        return;
    }
    
    // Allocate aligned memory
    float* aligned_data = static_cast<float*>(aligned_malloc(total_size, alignment));
    
    // Copy data to aligned location
    std::memcpy(aligned_data, data, total_size);
    
    // Copy back (in real implementation, we'd update pointers)
    std::memcpy(data, aligned_data, total_size);
    
    aligned_free(aligned_data);
}

void MemoryOptimizer::prefetch_memory(const float* data, size_t size) {
    const size_t prefetch_distance = 64; // Cache line size
    
    for (size_t i = 0; i < size; i += prefetch_distance / sizeof(float)) {
        _mm_prefetch(data + i, _MM_HINT_T0);
    }
}

// Parallel Optimizer Implementation
size_t ParallelOptimizer::get_optimal_thread_count() {
    size_t hardware_threads = std::thread::hardware_concurrency();
    
    // Consider memory bandwidth and cache effects
    // For vector operations, optimal is often less than hardware threads
    return std::min(hardware_threads, static_cast<size_t>(8));
}

void ParallelOptimizer::parallel_vector_copy(const float* src, float* dst, 
                                       size_t count, size_t dimension) {
    size_t num_threads = get_optimal_thread_count();
    size_t vectors_per_thread = (count + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_idx = t * vectors_per_thread;
        size_t end_idx = std::min(start_idx + vectors_per_thread, count);
        
        threads.emplace_back([src, dst, start_idx, end_idx, dimension]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                const float* src_vec = src + i * dimension;
                float* dst_vec = dst + i * dimension;
                std::memcpy(dst_vec, src_vec, dimension * sizeof(float));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

// Aligned memory allocation
void* aligned_malloc(size_t size, size_t alignment) {
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* raw = std::malloc(total_size);
    
    if (!raw) {
        return nullptr;
    }
    
    uintptr_t aligned = reinterpret_cast<uintptr_t>(raw) + sizeof(void*);
    aligned = (aligned + alignment - 1) & ~(alignment - 1);
    
    // Store original pointer for later free
    void** header = reinterpret_cast<void**>(aligned) - 1;
    *header = raw;
    
    return reinterpret_cast<void*>(aligned);
}

void aligned_free(void* ptr) {
    if (!ptr) {
        return;
    }
    
    void** header = reinterpret_cast<void**>(ptr) - 1;
    std::free(*header);
}

// CPU feature detection
bool has_avx2() {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[1] & (1 << 5)) != 0;
}

bool has_avx512() {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[1] & (1 << 16)) != 0;
}

bool has_fma() {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[1] & (1 << 12)) != 0;
}

bool has_sse4_2() {
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    return (cpu_info[2] & (1 << 20)) != 0;
}

} // namespace optimization
} // namespace hybridvectordb
