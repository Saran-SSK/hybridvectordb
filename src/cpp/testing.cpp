#include "benchmark.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

namespace hybridvectordb {

// AutomatedTestSuite Implementation
AutomatedTestSuite::AutomatedTestSuite() {
    // Initialize with basic tests
    add_test("Basic Functionality", [this]() {
        return test_basic_functionality();
    });
    
    add_test("Memory Management", [this]() {
        return test_memory_management();
    });
    
    add_test("Performance", [this]() {
        return test_performance();
    });
    
    add_test("Accuracy", [this]() {
        return test_accuracy();
    });
    
    add_test("Concurrency", [this]() {
        return test_concurrency();
    });
    
    add_test("Error Handling", [this]() {
        return test_error_handling();
    });
}

void AutomatedTestSuite::add_test(const std::string& name, std::function<bool()> test) {
    test_names_.push_back(name);
    tests_.push_back(test);
}

void AutomatedTestSuite::run_tests() {
    std::cout << "Running automated test suite..." << std::endl;
    
    test_results_.clear();
    test_times_.clear();
    
    for (size_t i = 0; i < tests_.size(); ++i) {
        std::cout << "\nRunning test: " << test_names_[i] << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool result = false;
        
        try {
            result = tests_[i]();
        } catch (const std::exception& e) {
            std::cout << "Test failed with exception: " << e.what() << std::endl;
            result = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        test_results_.push_back(result);
        test_times_.push_back(time_ms);
        
        std::cout << "Test " << (result ? "PASSED" : "FAILED") 
                  << " (" << time_ms << "ms)" << std::endl;
    }
    
    generate_test_report();
}

void AutomatedTestSuite::run_stress_tests() {
    std::cout << "\nRunning stress tests..." << std::endl;
    
    // High-load test
    add_test("High Load Test", [this]() {
        return stress_test_high_load();
    });
    
    // Memory stress test
    add_test("Memory Stress Test", [this]() {
        return stress_test_memory();
    });
    
    // Concurrency stress test
    add_test("Concurrency Stress Test", [this]() {
        return stress_test_concurrency();
    });
    
    // Long-running test
    add_test("Endurance Test", [this]() {
        return stress_test_endurance();
    });
    
    run_tests();
}

void AutomatedTestSuite::run_regression_tests() {
    std::cout << "\nRunning regression tests..." << std::endl;
    
    // Compare against baseline
    add_test("Regression - Performance", [this]() {
        return regression_test_performance();
    });
    
    add_test("Regression - Memory", [this]() {
        return regression_test_memory();
    });
    
    add_test("Regression - Accuracy", [this]() {
        return regression_test_accuracy();
    });
    
    run_tests();
}

void AutomatedTestSuite::generate_test_report() {
    std::cout << "\nGenerating test report..." << std::endl;
    
    std::ofstream file("test_report.html");
    file << R"(
<!DOCTYPE html>
<html>
<head>
    <title>HybridVectorDB Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .summary { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .test-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .test { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .test.passed { border-left: 4px solid #28a745; }
        .test.failed { border-left: 4px solid #dc3545; }
        .test-name { font-weight: bold; margin-bottom: 10px; }
        .test-status { font-size: 18px; margin-bottom: 10px; }
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .test-time { color: #666; font-size: 14px; }
        .coverage { background: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HybridVectorDB Test Report</h1>
            <p>Generated on: )";
    
    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    
    file << R"(</p>
        </div>
        
        <div class="summary">
            <h2>Test Summary</h2>
)";
    
    // Calculate summary
    size_t passed = std::count(test_results_.begin(), test_results_.end(), true);
    size_t failed = test_results_.size() - passed;
    double coverage = get_coverage();
    
    double avg_time = 0.0;
    if (!test_times_.empty()) {
        avg_time = std::accumulate(test_times_.begin(), test_times_.end(), 0.0) / test_times_.size();
    }
    
    file << R"(
            <div style="margin-bottom: 15px;">
                <strong>Total Tests:</strong> )" << test_results_.size() << R"(
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Passed:</strong> <span class="passed">)" << passed << R"(</span>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Failed:</strong> <span class="failed">)" << failed << R"(</span>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Success Rate:</strong> )" << std::fixed << std::setprecision(1) << (static_cast<double>(passed) / test_results_.size() * 100) << R"(%
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Average Test Time:</strong> )" << std::fixed << std::setprecision(2) << avg_time << R"(ms
            </div>
        </div>
        
        <div class="coverage">
            <h3>Test Coverage</h3>
            <div style="margin-bottom: 10px;">
                <strong>Estimated Coverage:</strong> )" << std::fixed << std::setprecision(1) << coverage << R"(%
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
)";
    
    for (size_t i = 0; i < test_results_.size(); ++i) {
        std::string status_class = test_results_[i] ? "passed" : "failed";
        std::string status_text = test_results_[i] ? "PASSED" : "FAILED";
        
        file << R"(
            <div class="test )" << status_class << R"(">
                <div class="test-name">)" << test_names_[i] << R"(</div>
                <div class="test-status )" << status_class << R"(">)" << status_text << R"(</div>
                <div class="test-time">)" << std::fixed << std::setprecision(2) << test_times_[i] << R"(ms</div>
            </div>
        )";
    }
    
    file << R"(
        </div>
    </div>
</body>
</html>
)";
    
    file.close();
    std::cout << "Test report exported to: test_report.html" << std::endl;
}

const std::vector<bool>& AutomatedTestSuite::get_results() const {
    return test_results_;
}

double AutomatedTestSuite::get_coverage() const {
    // Simple coverage estimation based on test types
    double coverage = 0.0;
    
    // Basic functionality tests (40%)
    if (std::find(test_names_.begin(), test_names_.end(), "Basic") != test_names_.end()) {
        coverage += 40.0;
    }
    
    // Memory management tests (20%)
    if (std::find(test_names_.begin(), test_names_.end(), "Memory") != test_names_.end()) {
        coverage += 20.0;
    }
    
    // Performance tests (20%)
    if (std::find(test_names_.begin(), test_names_.end(), "Performance") != test_names_.end()) {
        coverage += 20.0;
    }
    
    // Accuracy tests (15%)
    if (std::find(test_names_.begin(), test_names_.end(), "Accuracy") != test_names_.end()) {
        coverage += 15.0;
    }
    
    // Concurrency tests (5%)
    if (std::find(test_names_.begin(), test_names_.end(), "Concurrency") != test_names_.end()) {
        coverage += 5.0;
    }
    
    return coverage;
}

// Test implementations
bool AutomatedTestSuite::test_basic_functionality() {
    try {
        // Test basic vector operations
        Config config;
        config.dimension = 128;
        config.index_type = "flat";
        config.metric_type = "l2";
        
        auto db = create_vector_database(config);
        
        // Test add and search
        std::vector<std::vector<float>> vectors = {{1.0f, 2.0f, 3.0f}};
        auto added = db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        std::vector<float> query = {1.0f, 2.0f, 3.0f};
        auto results = db->search_vectors(query.data(), 1, 10, false);
        
        return added == 3 && results.size() == 1;
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::test_memory_management() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Test memory usage tracking
        auto memory_before = db->get_memory_usage();
        
        // Add large number of vectors
        std::vector<std::vector<float>> vectors(10000, std::vector<float>(128, 0.0f));
        db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        auto memory_after = db->get_memory_usage();
        
        // Check if memory usage increased
        size_t memory_before_bytes = std::stoull(memory_before["memory_pool_used"]);
        size_t memory_after_bytes = std::stoull(memory_after["memory_pool_used"]);
        
        return memory_after_bytes > memory_before_bytes;
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::test_performance() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Test performance metrics
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<float>> vectors(1000, std::vector<float>(128, 0.0f));
        db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        std::vector<float> query(128, 0.0f);
        auto results = db->search_vectors(query.data(), 1, 10, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Performance should be reasonable
        return time_ms < 100.0;  // Should complete in under 100ms
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::test_accuracy() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Create test data with known nearest neighbors
        std::vector<std::vector<float>> vectors = {
            {1.0f, 0.0f, 0.0f},  // Vector 0
            {0.0f, 1.0f, 0.0f},  // Vector 1 (closest to vector 0)
            {0.0f, 0.0f, 1.0f}   // Vector 2
        };
        
        // Add vectors
        for (size_t i = 0; i < vectors.size(); ++i) {
            VectorData vd("vec_" + std::to_string(i), vectors[i]);
            db->add_vectors(&vd, 1);
        }
        
        // Search for vector 0 - should return vector 0 as first result
        std::vector<float> query = {1.0f, 0.0f, 0.0f};
        auto results = db->search_vectors(query.data(), 1, 10, false);
        
        // Check if vector 0 is the first result (highest similarity)
        return !results.empty() && results[0].results[0].id == "vec_0";
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::test_concurrency() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Test concurrent operations
        std::vector<std::thread> threads;
        std::atomic<bool> all_passed{true};
        
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([this, &all_passed, i]() {
                try {
                    std::vector<std::vector<float>> vectors(100, std::vector<float>(128, 0.0f));
                    
                    for (size_t j = 0; j < vectors.size(); ++j) {
                        for (size_t k = 0; k < vectors[j].size(); ++k) {
                            vectors[j][k] = static_cast<float>(i * 1000 + j * 10 + k);
                        }
                    }
                    
                    VectorData vd("thread_" + std::to_string(i), vectors[0]);
                    auto added = db->add_vectors(&vd, vectors.size());
                    
                    if (added != vectors.size()) {
                        all_passed.store(false);
                    }
                } catch (...) {
                    all_passed.store(false);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return all_passed.load();
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::test_error_handling() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Test invalid inputs
        try {
            db->add_vectors(nullptr, 0);
            return false;  // Should not reach here
        } catch (...) {
            // Expected to throw
        }
        
        try {
            std::vector<float> wrong_dim(64, 0.0f);  // Wrong dimension
            VectorData vd("test", wrong_dim);
            db->add_vectors(&vd, 1);
            return false;  // Should not reach here
        } catch (...) {
            // Expected to throw
        }
        
        return true;  // Error handling worked correctly
    } catch (...) {
        return false;
    }
}

// Stress test implementations
bool AutomatedTestSuite::stress_test_high_load() {
    try {
        Config config;
        config.dimension = 256;  // Higher dimension for stress
        
        auto db = create_vector_database(config);
        
        // Add large number of vectors
        std::vector<std::vector<float>> vectors(50000, std::vector<float>(256, 0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        
        auto added = db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Should handle high load without crashing
        return added == vectors.size() && time_ms < 10000.0;  // 10 second limit
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::stress_test_memory() {
    try {
        Config config;
        config.dimension = 512;  // Large vectors
        
        auto db = create_vector_database(config);
        
        // Test memory limits
        std::vector<std::vector<float>> vectors;
        size_t added_count = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Keep adding until memory limit or failure
        while (added_count < 100000) {  // Reasonable limit
            std::vector<float> vector(512, static_cast<float>(added_count));
            VectorData vd("stress_" + std::to_string(added_count), vector);
            
            try {
                auto added = db->add_vectors(&vd, 1);
                if (added != 1) break;
                added_count += added;
            } catch (...) {
                break;
            }
            
            // Check memory usage periodically
            if (added_count % 10000 == 0) {
                auto memory_usage = db->get_memory_usage();
                size_t memory_mb = std::stoull(memory_usage["memory_pool_used"]) / 1024 / 1024;
                
                if (memory_mb > 1024) {  // 1GB limit
                    break;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Should handle memory stress gracefully
        return added_count > 1000 && time_ms < 30000.0;  // Added reasonable amount in reasonable time
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::stress_test_concurrency() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // High concurrency test
        std::vector<std::thread> threads;
        std::atomic<bool> all_passed{true};
        
        for (int i = 0; i < 16; ++i) {  // 16 concurrent threads
            threads.emplace_back([this, &all_passed, i]() {
                try {
                    for (int j = 0; j < 100; ++j) {  // 100 operations per thread
                        std::vector<float> query(128, static_cast<float>(i * 100 + j));
                        auto results = db->search_vectors(query.data(), 1, 10, false);
                        
                        if (results.empty()) {
                            all_passed.store(false);
                            break;
                        }
                    }
                } catch (...) {
                    all_passed.store(false);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return all_passed.load();
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::stress_test_endurance() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Long-running test
        auto start = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::seconds(30);  // 30 second endurance test
        
        std::atomic<bool> test_passed{true};
        std::thread test_thread([&]() {
            auto end = std::chrono::high_resolution_clock::now();
            while (std::chrono::high_resolution_clock::now() < start + duration) {
                try {
                    std::vector<float> query(128, 0.0f);
                    auto results = db->search_vectors(query.data(), 1, 10, false);
                    
                    if (results.empty()) {
                        test_passed.store(false);
                        break;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                } catch (...) {
                    test_passed.store(false);
                    break;
                }
            }
        });
        
        test_thread.join();
        
        return test_passed.load();
    } catch (...) {
        return false;
    }
}

// Regression test implementations
bool AutomatedTestSuite::regression_test_performance() {
    // Compare against baseline performance
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        // Standard performance test
        std::vector<std::vector<float>> vectors(1000, std::vector<float>(128, 0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        
        auto added = db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        std::vector<float> query(128, 0.0f);
        auto results = db->search_vectors(query.data(), 1, 10, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Performance should not regress significantly
        return added == vectors.size() && results.size() == 1 && time_ms < 50.0;  // Baseline expectation
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::regression_test_memory() {
    try {
        Config config;
        config.dimension = 128;
        
        auto db = create_vector_database(config);
        
        auto memory_before = db->get_memory_usage();
        
        std::vector<std::vector<float>> vectors(5000, std::vector<float>(128, 0.0f));
        auto added = db->add_vectors(
            reinterpret_cast<const VectorData*>(vectors.data()), 
            vectors.size()
        );
        
        auto memory_after = db->get_memory_usage();
        
        size_t memory_before_bytes = std::stoull(memory_before["memory_pool_used"]);
        size_t memory_after_bytes = std::stoull(memory_after["memory_pool_used"]);
        
        // Memory usage should be reasonable
        size_t expected_memory = vectors.size() * 128 * sizeof(float);
        double memory_ratio = static_cast<double>(memory_after_bytes) / expected_memory;
        
        return added == vectors.size() && memory_ratio < 3.0;  // Less than 3x expected memory
    } catch (...) {
        return false;
    }
}

bool AutomatedTestSuite::regression_test_accuracy() {
    try {
        Config config;
        config.dimension = 128;
        config.metric_type = "l2";
        
        auto db = create_vector_database(config);
        
        // Test with known data
        std::vector<std::vector<float>> vectors = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}
        };
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            VectorData vd("test_" + std::to_string(i), vectors[i]);
            db->add_vectors(&vd, 1);
        }
        
        // Search for first vector - should return itself as exact match
        std::vector<float> query = {1.0f, 0.0f, 0.0f};
        auto results = db->search_vectors(query.data(), 1, 10, false);
        
        // Should maintain accuracy
        return !results.empty() && results[0].results[0].id == "test_0" && 
               results[0].results[0].distance < 1e-6;
    } catch (...) {
        return false;
    }
}

} // namespace hybridvectordb
