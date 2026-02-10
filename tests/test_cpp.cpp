#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include "../src/cpp/vector_database.h"

using namespace hybridvectordb;

class HybridVectorDBTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.dimension = 128;
        config_.index_type = "flat";
        config_.metric_type = "l2";
        config_.use_gpu = false;
        config_.batch_threshold = 32;
        config_.k_threshold = 50;
        
        db_ = create_vector_database(config_);
        
        // Generate test data
        generate_test_data();
    }
    
    void TearDown() override {
        db_.reset();
    }
    
    void generate_test_data() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        test_vectors_.clear();
        test_vectors_.reserve(1000);
        
        for (size_t i = 0; i < 1000; ++i) {
            std::vector<float> embedding(config_.dimension);
            for (size_t j = 0; j < config_.dimension; ++j) {
                embedding[j] = dis(gen);
            }
            
            test_vectors_.emplace_back("vec_" + std::to_string(i), embedding);
        }
    }
    
    Config config_;
    std::unique_ptr<HybridVectorDB> db_;
    std::vector<VectorData> test_vectors_;
};

TEST_F(HybridVectorDBTest, InitializationTest) {
    EXPECT_NE(db_, nullptr);
    
    auto stats = db_->get_stats();
    EXPECT_EQ(stats["total_queries"], "0");
    EXPECT_EQ(stats["cpu_queries"], "0");
    EXPECT_EQ(stats["gpu_queries"], "0");
}

TEST_F(HybridVectorDBTest, AddVectorsTest) {
    // Add test vectors
    size_t added = db_->add_vectors(test_vectors_.data(), test_vectors_.size());
    EXPECT_EQ(added, test_vectors_.size());
    
    auto stats = db_->get_stats();
    EXPECT_NE(stats.find("total_vectors"), stats.end());
}

TEST_F(HybridVectorDBTest, SearchVectorsTest) {
    // Add vectors first
    db_->add_vectors(test_vectors_.data(), test_vectors_.size());
    
    // Prepare query
    std::vector<float> query(config_.dimension);
    for (size_t i = 0; i < config_.dimension; ++i) {
        query[i] = test_vectors_[0].embedding[i];
    }
    
    // Search
    auto results = db_->search_vectors(query.data(), 1, 10, false);
    EXPECT_EQ(results.size(), 1);
    EXPECT_LE(results[0].results.size(), 10);
    EXPECT_EQ(results[0].results[0].id, "vec_0");  // Should find exact match
    EXPECT_NEAR(results[0].results[0].distance, 0.0f, 1e-6f);
}

TEST_F(HybridVectorDBTest, ZeroCopyAccessTest) {
    // Add vectors
    db_->add_vectors(test_vectors_.data(), test_vectors_.size());
    
    // Zero-copy access
    size_t count;
    const float* vectors = db_->get_vectors_zero_copy(count);
    EXPECT_NE(vectors, nullptr);
    EXPECT_EQ(count, test_vectors_.size());
    
    // Verify data integrity
    for (size_t i = 0; i < std::min(count, test_vectors_.size()); ++i) {
        for (size_t j = 0; j < config_.dimension; ++j) {
            EXPECT_NEAR(vectors[i * config_.dimension + j], 
                       test_vectors_[i].embedding[j], 
                       1e-6f);
        }
    }
}

TEST_F(HybridVectorDBTest, PerformanceMetricsTest) {
    // Perform some operations
    db_->add_vectors(test_vectors_.data(), 100);
    
    std::vector<float> query(config_.dimension);
    for (size_t i = 0; i < config_.dimension; ++i) {
        query[i] = test_vectors_[0].embedding[i];
    }
    
    auto results = db_->search_vectors(query.data(), 1, 10, false);
    
    // Check metrics
    auto metrics = db_->get_metrics();
    EXPECT_GT(metrics.total_queries.load(), 0);
    EXPECT_GT(metrics.cpu_queries.load(), 0);
    EXPECT_GT(metrics.total_search_time_ms.load(), 0.0);
}

TEST_F(HybridVectorDBTest, PerformanceOptimizationTest) {
    // Add vectors
    db_->add_vectors(test_vectors_.data(), test_vectors_.size());
    
    // Optimize batch size
    auto optimization = db_->optimize_performance("batch_size");
    EXPECT_NE(optimization.find("optimal_batch_size"), optimization.end());
    EXPECT_NE(optimization.find("optimal_time_ms"), optimization.end());
    EXPECT_NE(optimization.find("throughput_vps"), optimization.end());
}

TEST_F(HybridVectorDBTest, MemoryUsageTest) {
    // Add vectors
    db_->add_vectors(test_vectors_.data(), test_vectors_.size());
    
    // Get memory usage
    auto memory_usage = db_->get_memory_usage();
    EXPECT_NE(memory_usage.find("memory_pool_capacity"), memory_usage.end());
    EXPECT_NE(memory_usage.find("memory_pool_used"), memory_usage.end());
    EXPECT_NE(memory_usage.find("memory_pool_free"), memory_usage.end());
}

TEST_F(HybridVectorDBTest, BenchmarkTest) {
    // Benchmark with different sizes
    auto benchmark_results = db_->benchmark(1000, 100, 10);
    
    EXPECT_NE(benchmark_results.find("num_vectors"), benchmark_results.end());
    EXPECT_NE(benchmark_results.find("num_queries"), benchmark_results.end());
    EXPECT_NE(benchmark_results.find("k"), benchmark_results.end());
    EXPECT_NE(benchmark_results.find("add_time_ms"), benchmark_results.end());
    EXPECT_NE(benchmark_results.find("avg_cpu_time_ms"), benchmark_results.end());
    EXPECT_NE(benchmark_results.find("throughput_qps_cpu"), benchmark_results.end());
}

TEST_F(HybridVectorDBTest, ErrorHandlingTest) {
    // Test invalid inputs
    EXPECT_THROW(db_->add_vectors(nullptr, 0), HybridVectorDBError);
    EXPECT_THROW(db_->search_vectors(nullptr, 0, 10, false), HybridVectorDBError);
    
    // Test dimension mismatch
    std::vector<float> wrong_dim_vector(64, 0.0f);
    std::vector<VectorData> wrong_dim_data;
    wrong_dim_data.emplace_back("wrong", wrong_dim_vector);
    
    EXPECT_THROW(db_->add_vectors(wrong_dim_data.data(), wrong_dim_data.size()), 
                 HybridVectorDBError);
}

TEST_F(HybridVectorDBTest, ConfigurationTest) {
    // Test configuration changes
    Config new_config = config_;
    new_config.batch_threshold = 64;
    new_config.k_threshold = 100;
    
    db_->configure(new_config);
    
    auto stats = db_->get_stats();
    // Config change should be reflected in subsequent operations
    EXPECT_NO_THROW(db_->add_vectors(test_vectors_.data(), 10));
}

// Performance benchmarks
class PerformanceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        config_.dimension = 128;
        config_.index_type = "flat";
        config_.metric_type = "l2";
        config_.use_gpu = false;
        
        db_ = create_vector_database(config_);
        
        // Generate large dataset
        generate_large_dataset();
    }
    
    void generate_large_dataset() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        large_vectors_.clear();
        large_vectors_.reserve(10000);
        
        for (size_t i = 0; i < 10000; ++i) {
            std::vector<float> embedding(config_.dimension);
            for (size_t j = 0; j < config_.dimension; ++j) {
                embedding[j] = dis(gen);
            }
            
            large_vectors_.emplace_back("vec_" + std::to_string(i), embedding);
        }
    }
    
    Config config_;
    std::unique_ptr<HybridVectorDB> db_;
    std::vector<VectorData> large_vectors_;
};

TEST_F(PerformanceBenchmark, AddPerformanceTest) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t added = db_->add_vectors(large_vectors_.data(), large_vectors_.size());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_EQ(added, large_vectors_.size());
    
    double vectors_per_second = (added * 1000000.0) / duration.count();
    EXPECT_GT(vectors_per_second, 1000.0);  // Should add at least 1000 vectors/sec
    
    std::cout << "Add performance: " << vectors_per_second << " vectors/second" << std::endl;
}

TEST_F(PerformanceBenchmark, SearchPerformanceTest) {
    // Add vectors first
    db_->add_vectors(large_vectors_.data(), large_vectors_.size());
    
    // Prepare queries
    std::vector<std::vector<float>> queries(1000);
    for (size_t i = 0; i < 1000; ++i) {
        queries[i].resize(config_.dimension);
        for (size_t j = 0; j < config_.dimension; ++j) {
            queries[i][j] = large_vectors_[i % large_vectors_.size()].embedding[j];
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& query : queries) {
        auto results = db_->search_vectors(query.data(), 1, 10, false);
        EXPECT_EQ(results.size(), 1);
        EXPECT_LE(results[0].results.size(), 10);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double queries_per_second = (1000 * 1000000.0) / duration.count();
    EXPECT_GT(queries_per_second, 100.0);  // Should handle at least 100 queries/sec
    
    std::cout << "Search performance: " << queries_per_second << " queries/second" << std::endl;
}

TEST_F(PerformanceBenchmark, MemoryEfficiencyTest) {
    // Add vectors
    db_->add_vectors(large_vectors_.data(), large_vectors_.size());
    
    auto memory_usage = db_->get_memory_usage();
    
    size_t expected_memory = large_vectors_.size() * config_.dimension * sizeof(float);
    size_t actual_memory = std::stoull(memory_usage["memory_pool_used"]);
    
    // Memory usage should be reasonable (within 2x of expected)
    EXPECT_LT(actual_memory, expected_memory * 2);
    
    double efficiency = (double)expected_memory / actual_memory * 100.0;
    EXPECT_GT(efficiency, 50.0);  // At least 50% efficiency
    
    std::cout << "Memory efficiency: " << efficiency << "%" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
