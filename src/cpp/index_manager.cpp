#include "index_manager.h"
#include <stdexcept>
#include <iostream>

namespace hybridvectordb {

IndexManager::IndexManager(const Config& config) : config_(config) {
    initialize_index();
}

IndexManager::~IndexManager() {
    cleanup_index();
}

void IndexManager::initialize_index() {
    try {
        if (config_.use_gpu) {
            initialize_gpu_index();
        } else {
            initialize_cpu_index();
        }
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to initialize index: " + std::string(e.what()), "INDEX_INIT_ERROR");
    }
}

void IndexManager::initialize_cpu_index() {
    // Initialize FAISS CPU index
    if (config_.index_type == "flat") {
        cpu_index_ = std::make_unique<faiss::IndexFlatL2>(config_.dimension);
    } else if (config_.index_type == "ivf") {
        auto quantizer = std::make_unique<faiss::IndexFlatL2>(config_.dimension);
        cpu_index_ = std::make_unique<faiss::IndexIVFFlat>(
            quantizer.release(), config_.dimension, config_.nlist
        );
    } else {
        throw HybridVectorDBError("Unsupported index type: " + config_.index_type, "INVALID_INDEX_TYPE");
    }
    
    std::cout << "CPU index initialized: " << config_.index_type << std::endl;
}

void IndexManager::initialize_gpu_index() {
    // Initialize FAISS GPU index
    try {
        int device = 0;
        faiss::gpu::StandardGpuResources* gpu_res = new faiss::gpu::StandardGpuResources();
        
        if (config_.index_type == "flat") {
            gpu_index_ = faiss::gpu::index_cpu_to_gpu(gpu_res, device, cpu_index_.get());
        } else if (config_.index_type == "ivf") {
            gpu_index_ = faiss::gpu::index_cpu_to_gpu(gpu_res, device, cpu_index_.get());
        }
        
        std::cout << "GPU index initialized: " << config_.index_type << std::endl;
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to initialize GPU index: " + std::string(e.what()), "GPU_INDEX_ERROR");
    }
}

void IndexManager::cleanup_index() {
    cpu_index_.reset();
    gpu_index_.reset();
}

void IndexManager::configure(const Config& config) {
    config_ = config;
    cleanup_index();
    initialize_index();
}

size_t IndexManager::add_vectors(const float* vectors, size_t count) {
    if (!vectors || count == 0) {
        return 0;
    }
    
    try {
        faiss::Index* index = get_active_index();
        index->add(count, vectors);
        return count;
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to add vectors: " + std::string(e.what()), "ADD_VECTORS_ERROR");
    }
}

void IndexManager::search_vectors(const float* query, size_t query_count, 
                                 size_t k, float* distances, int64_t* labels) {
    if (!query || query_count == 0 || k == 0) {
        return;
    }
    
    try {
        faiss::Index* index = get_active_index();
        index->search(query_count, query, k, distances, labels);
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to search vectors: " + std::string(e.what()), "SEARCH_VECTORS_ERROR");
    }
}

size_t IndexManager::get_vector_count() const {
    try {
        faiss::Index* index = get_active_index();
        return index->ntotal;
    } catch (const std::exception& e) {
        return 0;
    }
}

std::vector<float> IndexManager::get_vectors_zero_copy() {
    // This is a simplified implementation
    // In a real implementation, this would return direct memory access
    std::vector<float> empty_vector;
    return empty_vector;
}

std::map<std::string, std::string> IndexManager::get_memory_usage() const {
    std::map<std::string, std::string> memory_info;
    
    try {
        faiss::Index* index = get_active_index();
        
        // Get memory usage (simplified)
        size_t index_size = index->ntotal * index->d * sizeof(float);
        
        memory_info["index_size_bytes"] = std::to_string(index_size);
        memory_info["index_size_mb"] = std::to_string(index_size / (1024 * 1024));
        memory_info["vector_count"] = std::to_string(index->ntotal);
        memory_info["dimension"] = std::to_string(index->d);
        
    } catch (const std::exception& e) {
        memory_info["error"] = e.what();
    }
    
    return memory_info;
}

faiss::Index* IndexManager::get_active_index() const {
    if (config_.use_gpu && gpu_index_) {
        return gpu_index_.get();
    } else if (cpu_index_) {
        return cpu_index_.get();
    } else {
        throw HybridVectorDBError("No index available", "NO_INDEX_ERROR");
    }
}

void IndexManager::save_index(const std::string& filepath) {
    try {
        faiss::Index* index = get_active_index();
        faiss::write_index(index, filepath.c_str());
        std::cout << "Index saved to: " << filepath << std::endl;
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to save index: " + std::string(e.what()), "SAVE_INDEX_ERROR");
    }
}

void IndexManager::load_index(const std::string& filepath) {
    try {
        faiss::Index* loaded_index = faiss::read_index(filepath.c_str());
        
        // Replace current index
        if (config_.use_gpu && gpu_index_) {
            gpu_index_.reset(loaded_index);
        } else {
            cpu_index_.reset(loaded_index);
        }
        
        std::cout << "Index loaded from: " << filepath << std::endl;
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to load index: " + std::string(e.what()), "LOAD_INDEX_ERROR");
    }
}

void IndexManager::train_index(const float* training_vectors, size_t training_size) {
    try {
        faiss::Index* index = get_active_index();
        
        if (index->get_index_type() == faiss::IndexType::IndexIVFFlat) {
            dynamic_cast<faiss::IndexIVFFlat*>(index)->train(training_size, training_vectors);
            std::cout << "Index trained with " << training_size << " vectors" << std::endl;
        }
    } catch (const std::exception& e) {
        throw HybridVectorDBError("Failed to train index: " + std::string(e.what()), "TRAIN_INDEX_ERROR");
    }
}

} // namespace hybridvectordb
