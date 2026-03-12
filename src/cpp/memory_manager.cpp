#include "memory_manager.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace hybridvectordb {

MemoryManager::MemoryManager(size_t pool_size) : pool_size_(pool_size), allocated_bytes_(0) {
    initialize_memory_pool();
}

MemoryManager::~MemoryManager() {
    cleanup_memory_pool();
}

void MemoryManager::initialize_memory_pool() {
    try {
        memory_pool_ = std::make_unique<HighPerformanceMemoryPool>(pool_size_);
        std::cout << "Memory pool initialized: " << pool_size_ / (1024 * 1024) << " MB" << std::endl;
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to initialize memory pool: " + std::string(e.what()), "MEMORY_POOL_ERROR");
    }
}

void MemoryManager::cleanup_memory_pool() {
    memory_pool_.reset();
    allocated_bytes_ = 0;
}

void* MemoryManager::allocate_aligned(size_t size, size_t alignment) {
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    
    if (!ptr) {
        throw HybridVectorDBError("Failed to allocate aligned memory", "ALLOCATION_ERROR");
    }
    
    allocated_bytes_ += size;
    return ptr;
}

void MemoryManager::deallocate_aligned(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

void* MemoryManager::allocate_from_pool(size_t size) {
    if (memory_pool_) {
        return memory_pool_->allocate(size);
    }
    return allocate_aligned(size, 32);
}

void MemoryManager::deallocate_to_pool(void* ptr, size_t size) {
    if (memory_pool_) {
        memory_pool_->deallocate(ptr, size);
    } else {
        deallocate_aligned(ptr);
    }
}

std::vector<MemoryBlock> MemoryManager::get_memory_blocks() const {
    std::vector<MemoryBlock> blocks;
    
    if (memory_pool_) {
        // Return memory blocks from the pool
        // This is a simplified implementation
        MemoryBlock block;
        block.size = pool_size_;
        block.allocated = allocated_bytes_;
        block.free = pool_size_ - allocated_bytes_;
        blocks.push_back(block);
    }
    
    return blocks;
}

size_t MemoryManager::get_allocated_bytes() const {
    return allocated_bytes_;
}

size_t MemoryManager::get_free_bytes() const {
    return pool_size_ - allocated_bytes_;
}

double MemoryManager::get_memory_utilization() const {
    if (pool_size_ == 0) return 0.0;
    return static_cast<double>(allocated_bytes_) / pool_size_;
}

void MemoryManager::optimize_memory_layout(float* data, size_t rows, size_t cols) {
    if (!data || rows == 0 || cols == 0) {
        return;
    }
    
    // Optimize for cache performance
    // This is a simplified implementation
    size_t block_size = 64; // Cache line size
    size_t elements_per_block = block_size / sizeof(float);
    
    for (size_t i = 0; i < rows; i += elements_per_block) {
        size_t end_row = std::min(i + elements_per_block, rows);
        
        // Process in cache-friendly blocks
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = i; k < end_row; ++k) {
                // Prefetch next cache line
                if (k + elements_per_block < rows) {
                    memory_ops::prefetch_memory(&data[(k + elements_per_block) * cols + j]);
                }
            }
        }
    }
}

void MemoryManager::prefetch_memory(const void* addr) {
    memory_ops::prefetch_memory(addr);
}

// HighPerformanceMemoryPool Implementation
HighPerformanceMemoryPool::HighPerformanceMemoryPool(size_t pool_size) 
    : pool_size_(pool_size), allocated_(0) {
    
    // Allocate the memory pool
    pool_memory_ = allocate_aligned(pool_size_, 32);
    if (!pool_memory_) {
        throw HybridVectorDBError("Failed to allocate memory pool", "POOL_ALLOCATION_ERROR");
    }
    
    // Initialize free list
    free_blocks_.push_back({pool_memory_, pool_size_});
}

HighPerformanceMemoryPool::~HighPerformanceMemoryPool() {
    if (pool_memory_) {
        deallocate_aligned(pool_memory_);
    }
}

void* HighPerformanceMemoryPool::allocate(size_t size) {
    // Find a suitable free block
    auto it = std::find_if(free_blocks_.begin(), free_blocks_.end(),
        [size](const MemoryBlockInfo& block) {
            return block.size >= size;
        });
    
    if (it == free_blocks_.end()) {
        throw HybridVectorDBError("Out of memory in pool", "POOL_OUT_OF_MEMORY");
    }
    
    void* ptr = it->ptr;
    size_t block_size = it->size;
    
    // Remove from free list
    free_blocks_.erase(it);
    
    // Add allocated block
    allocated_blocks_[ptr] = {ptr, size};
    
    // If there's remaining space, add it back to free list
    if (block_size > size) {
        void* remaining_ptr = static_cast<char*>(ptr) + size;
        size_t remaining_size = block_size - size;
        free_blocks_.push_back({remaining_ptr, remaining_size});
    }
    
    allocated_ += size;
    return ptr;
}

void HighPerformanceMemoryPool::deallocate(void* ptr, size_t size) {
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        return; // Not allocated from this pool
    }
    
    // Remove from allocated blocks
    allocated_blocks_.erase(it);
    
    // Add back to free list
    free_blocks_.push_back({ptr, size});
    allocated_ -= size;
    
    // Try to coalesce adjacent free blocks
    coalesce_free_blocks();
}

void HighPerformanceMemoryPool::coalesce_free_blocks() {
    if (free_blocks_.size() < 2) {
        return;
    }
    
    // Sort by address
    std::sort(free_blocks_.begin(), free_blocks_.end(),
        [](const MemoryBlockInfo& a, const MemoryBlockInfo& b) {
            return a.ptr < b.ptr;
        });
    
    // Coalesce adjacent blocks
    for (size_t i = 0; i < free_blocks_.size() - 1; ++i) {
        char* current_end = static_cast<char*>(free_blocks_[i].ptr) + free_blocks_[i].size;
        if (current_end == free_blocks_[i + 1].ptr) {
            // Coalesce these blocks
            free_blocks_[i].size += free_blocks_[i + 1].size;
            free_blocks_.erase(free_blocks_.begin() + i + 1);
            --i; // Check again from this position
        }
    }
}

// memory_ops namespace implementation
namespace memory_ops {

void* allocate_aligned(size_t size, size_t alignment) {
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    
    return ptr;
}

void deallocate_aligned(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

void prefetch_memory(const void* addr) {
    if (addr) {
#ifdef _MSC_VER
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
        __builtin_prefetch(addr, 0, 3);
#endif
    }
}

void copy_memory(void* dest, const void* src, size_t size) {
    std::memcpy(dest, src, size);
}

void set_memory(void* dest, int value, size_t size) {
    std::memset(dest, value, size);
}

} // namespace memory_ops

} // namespace hybridvectordb
