#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <atomic>
#include <mutex>

namespace hybridvectordb {

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig {
    size_t num_vectors = 10000;
    size_t num_queries = 1000;
    size_t vector_dimension = 128;
    size_t k_values[] = {10, 50, 100, 500};
    std::vector<size_t> batch_sizes = {1, 10, 50, 100, 500, 1000};
    std::vector<std::string> index_types = {"flat", "ivf"};
    std::vector<std::string> metrics = {"l2", "inner_product", "cosine"};
    bool use_gpu = false;
    bool use_cpp = true;
    bool enable_profiling = true;
    size_t warmup_iterations = 10;
    size_t benchmark_iterations = 5;
    std::string output_format = "json";  // "json", "csv", "html"
    std::string output_file = "benchmark_results";
};

/**
 * @brief Benchmark metrics
 */
struct BenchmarkMetrics {
    // Performance metrics
    double add_time_ms = 0.0;
    double search_time_ms = 0.0;
    double total_time_ms = 0.0;
    size_t vectors_added = 0;
    size_t queries_processed = 0;
    
    // Throughput metrics
    double add_throughput_vps = 0.0;  // vectors per second
    double search_throughput_qps = 0.0;  // queries per second
    double overall_throughput_ops = 0.0;  // operations per second
    
    // Memory metrics
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;
    double memory_efficiency = 0.0;  // percentage
    
    // Accuracy metrics
    double recall_at_k = 0.0;
    double precision_at_k = 0.0;
    double f1_score = 0.0;
    
    // Resource utilization
    double cpu_utilization = 0.0;
    double gpu_utilization = 0.0;
    double memory_utilization = 0.0;
    
    // Statistical metrics
    double avg_query_time_ms = 0.0;
    double min_query_time_ms = 0.0;
    double max_query_time_ms = 0.0;
    double std_deviation_ms = 0.0;
    double percentile_95_ms = 0.0;
    double percentile_99_ms = 0.0;
    
    // Comparison metrics
    double python_vs_cpp_speedup = 0.0;
    double cpu_vs_gpu_speedup = 0.0;
    double baseline_speedup = 0.0;
};

/**
 * @brief Load generation configuration
 */
struct LoadConfig {
    size_t num_vectors = 100000;
    size_t vector_dimension = 128;
    std::string distribution = "uniform";  // "uniform", "normal", "exponential", "clustered"
    double cluster_std = 0.1;
    size_t num_clusters = 10;
    double noise_ratio = 0.05;
    bool normalize_vectors = true;
    std::string output_format = "numpy";  // "numpy", "faiss", "hdf5"
    std::string output_file = "generated_vectors";
};

/**
 * @brief Comprehensive benchmarking framework
 */
class BenchmarkFramework {
private:
    BenchmarkConfig config_;
    std::unique_ptr<class HybridVectorDB> db_;
    std::unique_ptr<class HybridVectorDB> python_db_;
    std::vector<BenchmarkMetrics> results_;
    std::atomic<bool> running_{false};
    std::mutex results_mutex_;
    
    // Performance monitoring
    struct PerformanceMonitor {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        size_t memory_before;
        size_t memory_after;
        std::unordered_map<std::string, double> custom_metrics;
    };
    
    std::vector<PerformanceMonitor> monitors_;
    
public:
    explicit BenchmarkFramework(const BenchmarkConfig& config);
    ~BenchmarkFramework();
    
    /**
     * @brief Run comprehensive benchmark
     */
    void run_benchmark();
    
    /**
     * @brief Generate benchmark report
     */
    void generate_report();
    
    /**
     * @brief Get benchmark results
     */
    const std::vector<BenchmarkMetrics>& get_results() const;
    
    /**
     * @brief Export results to file
     */
    void export_results(const std::string& filename);
    
    /**
     * @brief Visualize results
     */
    void visualize_results();
    
private:
    /**
     * @brief Initialize databases
     */
    void initialize_databases();
    
    /**
     * @brief Generate test data
     */
    std::vector<std::vector<float>> generate_test_data(size_t count);
    
    /**
     * @brief Run add benchmark
     */
    BenchmarkMetrics benchmark_add_operations();
    
    /**
     * @brief Run search benchmark
     */
    BenchmarkMetrics benchmark_search_operations();
    
    /**
     * @brief Run throughput benchmark
     */
    BenchmarkMetrics benchmark_throughput();
    
    /**
     * @brief Run memory benchmark
     */
    BenchmarkMetrics benchmark_memory_usage();
    
    /**
     * @brief Run accuracy benchmark
     */
    BenchmarkMetrics benchmark_accuracy();
    
    /**
     * @brief Compare implementations
     */
    void compare_implementations();
    
    /**
     * @brief Calculate statistical metrics
     */
    void calculate_statistics(std::vector<double>& values, BenchmarkMetrics& metrics);
    
    /**
     * @brief Monitor system resources
     */
    void start_monitoring(const std::string& operation);
    void stop_monitoring(const std::string& operation, BenchmarkMetrics& metrics);
};

/**
 * @brief Load generation utilities
 */
class LoadGenerator {
private:
    LoadConfig config_;
    
public:
    explicit LoadGenerator(const LoadConfig& config);
    
    /**
     * @brief Generate synthetic vectors
     */
    std::vector<std::vector<float>> generate_vectors();
    
    /**
     * @brief Generate clustered vectors
     */
    std::vector<std::vector<float>> generate_clustered_vectors();
    
    /**
     * @brief Generate real-world like data
     */
    std::vector<std::vector<float>> generate_realistic_vectors();
    
    /**
     * @brief Add noise to vectors
     */
    void add_noise(std::vector<std::vector<float>>& vectors);
    
    /**
     * @brief Normalize vectors
     */
    void normalize_vectors(std::vector<std::vector<float>>& vectors);
    
    /**
     * @brief Export generated data
     */
    void export_data(const std::vector<std::vector<float>>& vectors);
    
    /**
     * @brief Validate generated data
     */
    bool validate_data(const std::vector<std::vector<float>>& vectors);
};

/**
 * @brief Performance visualization
 */
class PerformanceVisualizer {
private:
    std::vector<BenchmarkMetrics> data_;
    std::string output_format_;
    
public:
    explicit PerformanceVisualizer(const std::vector<BenchmarkMetrics>& data, 
                             const std::string& format = "html");
    
    /**
     * @brief Generate performance dashboard
     */
    void generate_dashboard();
    
    /**
     * @brief Generate performance charts
     */
    void generate_charts();
    
    /**
     * @brief Generate comparison plots
     */
    void generate_comparison_plots();
    
    /**
     * @brief Generate HTML report
     */
    void generate_html_report();
    
    /**
     * @brief Export visualization
     */
    void export_visualization(const std::string& filename);
    
private:
    /**
     * @brief Create throughput chart
     */
    void create_throughput_chart();
    
    /**
     * @brief Create latency chart
     */
    void create_latency_chart();
    
    /**
     * @brief Create memory usage chart
     */
    void create_memory_chart();
    
    /**
     * @brief Create accuracy chart
     */
    void create_accuracy_chart();
    
    /**
     * @brief Create comparison table
     */
    void create_comparison_table();
};

/**
 * @brief Automated testing suite
 */
class AutomatedTestSuite {
private:
    std::vector<std::function<bool()>> tests_;
    std::vector<std::string> test_names_;
    std::vector<bool> test_results_;
    std::vector<double> test_times_;
    
public:
    AutomatedTestSuite();
    
    /**
     * @brief Add test to suite
     */
    void add_test(const std::string& name, std::function<bool()> test);
    
    /**
     * @brief Run all tests
     */
    void run_tests();
    
    /**
     * @brief Run stress tests
     */
    void run_stress_tests();
    
    /**
     * @brief Run regression tests
     */
    void run_regression_tests();
    
    /**
     * @brief Generate test report
     */
    void generate_test_report();
    
    /**
     * @brief Get test results
     */
    const std::vector<bool>& get_results() const;
    
    /**
     * @brief Get test coverage
     */
    double get_coverage() const;
};

/**
 * @brief Benchmark report generator
 */
class BenchmarkReport {
private:
    std::vector<BenchmarkMetrics> results_;
    std::string template_path_;
    
public:
    explicit BenchmarkReport(const std::vector<BenchmarkMetrics>& results);
    
    /**
     * @brief Generate comprehensive report
     */
    void generate_comprehensive_report();
    
    /**
     * @brief Generate executive summary
     */
    void generate_executive_summary();
    
    /**
     * @brief Generate detailed analysis
     */
    void generate_detailed_analysis();
    
    /**
     * @brief Generate recommendations
     */
    void generate_recommendations();
    
    /**
     * @brief Export report
     */
    void export_report(const std::string& format, const std::string& filename);
    
private:
    /**
     * @brief Format metrics for display
     */
    std::string format_metric(double value, const std::string& unit);
    std::string format_size(size_t bytes);
    std::string format_throughput(double ops_per_sec);
    
    /**
     * @brief Generate markdown table
     */
    std::string generate_markdown_table();
    
    /**
     * @brief Generate HTML table
     */
    std::string generate_html_table();
};

} // namespace hybridvectordb
