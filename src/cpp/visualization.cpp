#include "benchmark.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <numeric>

namespace hybridvectordb {

// PerformanceVisualizer Implementation
PerformanceVisualizer::PerformanceVisualizer(const std::vector<BenchmarkMetrics>& data, 
                                       const std::string& format)
    : data_(data), output_format_(format) {}

void PerformanceVisualizer::generate_dashboard() {
    std::cout << "Generating performance dashboard..." << std::endl;
    
    // Calculate summary statistics
    double avg_add_time = 0.0, avg_search_time = 0.0;
    double max_throughput = 0.0, min_throughput = std::numeric_limits<double>::max();
    
    for (const auto& m : data_) {
        avg_add_time += m.add_time_ms;
        avg_search_time += m.search_time_ms;
        max_throughput = std::max(max_throughput, m.overall_throughput_ops);
        min_throughput = std::min(min_throughput, m.overall_throughput_ops);
    }
    
    if (!data_.empty()) {
        avg_add_time /= data_.size();
        avg_search_time /= data_.size();
    }
    
    // Generate HTML dashboard
    std::stringstream html;
    html << R"(
<!DOCTYPE html>
<html>
<head>
    <title>HybridVectorDB Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric { font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px; }
        .submetric { font-size: 14px; color: #666; margin-bottom: 5px; }
        .chart { height: 300px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>HybridVectorDB Performance Dashboard</h1>
    
    <div class="dashboard">
        <div class="card">
            <div class="metric">)"" << std::fixed << std::setprecision(1) << avg_add_time << R"(ms</div>
            <div class="submetric">Average Add Time</div>
        </div>
        
        <div class="card">
            <div class="metric">)"" << std::fixed << std::setprecision(1) << avg_search_time << R"(ms</div>
            <div class="submetric">Average Search Time</div>
        </div>
        
        <div class="card">
            <div class="metric">)"" << std::fixed << std::setprecision(0) << max_throughput << R"( ops/s</div>
            <div class="submetric">Peak Throughput</div>
        </div>
        
        <div class="card">
            <div class="metric">)"" << std::fixed << std::setprecision(1) << min_throughput << R"( ops/s</div>
            <div class="submetric">Min Throughput</div>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <canvas id="throughputChart" class="chart"></canvas>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <canvas id="latencyChart" class="chart"></canvas>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <canvas id="memoryChart" class="chart"></canvas>
        </div>
    </div>
</body>

<script>
    // Throughput Chart
    const throughputCtx = document.getElementById('throughputChart').getContext('2d');
    const throughputChart = new Chart(throughputCtx, {
        type: 'line',
        data: {
            labels: [)";
    
    // Add data points
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << (i + 1);
    }
    
    html << R"(],
            datasets: [{
                label: 'Overall Throughput',
                data: [)";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << std::fixed << std::setprecision(1) << data_[i].overall_throughput_ops;
    }
    
    html << R"(],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }]
        }
    });
    
    // Latency Chart
    const latencyCtx = document.getElementById('latencyChart').getContext('2d');
    const latencyChart = new Chart(latencyCtx, {
        type: 'line',
        data: {
            labels: [)";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << (i + 1);
    }
    
    html << R"(],
            datasets: [{
                label: 'Search Latency',
                data: [)";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << std::fixed << std::setprecision(2) << data_[i].search_time_ms;
    }
    
    html << R"(],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1
            }]
        }
    });
    
    // Memory Chart
    const memoryCtx = document.getElementById('memoryChart').getContext('2d');
    const memoryChart = new Chart(memoryCtx, {
        type: 'bar',
        data: {
            labels: [)";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << (i + 1);
    }
    
    html << R"(],
            datasets: [{
                label: 'Memory Usage (MB)',
                data: [)";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) html << ",";
        html << std::fixed << std::setprecision(1) << (data_[i].memory_usage_bytes / 1024.0 / 1024.0);
    }
    
    html << R"(],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        }
    });
    
</script>
</html>
)";
    
    // Write to file
    std::ofstream file("dashboard.html");
    file << html.str();
    file.close();
}

void PerformanceVisualizer::generate_charts() {
    std::cout << "Generating performance charts..." << std::endl;
    
    // Generate individual chart files
    create_throughput_chart();
    create_latency_chart();
    create_memory_chart();
    create_accuracy_chart();
}

void PerformanceVisualizer::create_throughput_chart() {
    std::stringstream chart_data;
    chart_data << "iteration,add_throughput,search_throughput,overall_throughput\n";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        chart_data << (i + 1) << ","
                   << m.add_throughput_vps << ","
                   << m.search_throughput_qps << ","
                   << m.overall_throughput_ops << "\n";
    }
    
    std::ofstream file("throughput_chart.csv");
    file << chart_data.str();
    file.close();
}

void PerformanceVisualizer::create_latency_chart() {
    std::stringstream chart_data;
    chart_data << "iteration,add_time_ms,search_time_ms,avg_query_time_ms\n";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        chart_data << (i + 1) << ","
                   << m.add_time_ms << ","
                   << m.search_time_ms << ","
                   << m.avg_query_time_ms << "\n";
    }
    
    std::ofstream file("latency_chart.csv");
    file << chart_data.str();
    file.close();
}

void PerformanceVisualizer::create_memory_chart() {
    std::stringstream chart_data;
    chart_data << "iteration,memory_usage_mb,peak_memory_mb,memory_efficiency\n";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        chart_data << (i + 1) << ","
                   << (m.memory_usage_bytes / 1024.0 / 1024.0) << ","
                   << (m.peak_memory_bytes / 1024.0 / 1024.0) << ","
                   << m.memory_efficiency << "\n";
    }
    
    std::ofstream file("memory_chart.csv");
    file << chart_data.str();
    file.close();
}

void PerformanceVisualizer::create_accuracy_chart() {
    std::stringstream chart_data;
    chart_data << "iteration,recall_at_k,precision_at_k,f1_score\n";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        chart_data << (i + 1) << ","
                   << m.recall_at_k << ","
                   << m.precision_at_k << ","
                   << m.f1_score << "\n";
    }
    
    std::ofstream file("accuracy_chart.csv");
    file << chart_data.str();
    file.close();
}

void PerformanceVisualizer::create_comparison_table() {
    std::cout << "\nPerformance Comparison Table:\n";
    std::cout << std::left << std::setw(15) << "Iteration" 
                 << std::setw(15) << "Add Time (ms)" 
                 << std::setw(15) << "Search Time (ms)" 
                 << std::setw(15) << "Throughput (ops/s)" 
                 << std::setw(15) << "Memory (MB)" 
                 << std::setw(15) << "Recall" << std::endl;
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        std::cout << std::left << std::setw(15) << (i + 1)
                     << std::setw(15) << std::fixed << std::setprecision(2) << m.add_time_ms
                     << std::setw(15) << std::fixed << std::setprecision(2) << m.search_time_ms
                     << std::setw(15) << std::fixed << std::setprecision(0) << m.overall_throughput_ops
                     << std::setw(15) << std::fixed << std::setprecision(1) << (m.memory_usage_bytes / 1024.0 / 1024.0)
                     << std::setw(15) << std::fixed << std::setprecision(3) << m.recall_at_k << std::endl;
    }
}

void PerformanceVisualizer::generate_html_report() {
    std::cout << "Generating HTML report..." << std::endl;
    
    std::stringstream html;
    html << R"(
<!DOCTYPE html>
<html>
<head>
    <title>HybridVectorDB Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .section { margin-bottom: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric h3 { margin-top: 0; color: #333; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .summary { background: #e9ecef; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HybridVectorDB Performance Report</h1>
            <p>Generated on: )";
    
    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    html << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    
    html << R"(</p>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric-grid">)";
    
    // Calculate summary metrics
    if (!data_.empty()) {
        double total_add_time = 0, total_search_time = 0;
        double avg_throughput = 0, max_memory = 0;
        
        for (const auto& m : data_) {
            total_add_time += m.add_time_ms;
            total_search_time += m.search_time_ms;
            avg_throughput += m.overall_throughput_ops;
            max_memory = std::max(max_memory, static_cast<double>(m.memory_usage_bytes));
        }
        
        total_add_time /= data_.size();
        total_search_time /= data_.size();
        avg_throughput /= data_.size();
        
        html << R"(
                <div class="metric">
                    <h3>Average Add Time</h3>
                    <div class="metric-value">)" << std::fixed << std::setprecision(2) << total_add_time << R"(ms</div>
                    <div class="metric-label">Across )" << data_.size() << R"( iterations</div>
                </div>
                <div class="metric">
                    <h3>Average Search Time</h3>
                    <div class="metric-value">)" << std::fixed << std::setprecision(2) << total_search_time << R"(ms</div>
                    <div class="metric-label">Across )" << data_.size() << R"( iterations</div>
                </div>
                <div class="metric">
                    <h3>Average Throughput</h3>
                    <div class="metric-value">)" << std::fixed << std::setprecision(0) << avg_throughput << R"( ops/s</div>
                    <div class="metric-label">Operations per second</div>
                </div>
                <div class="metric">
                    <h3>Peak Memory Usage</h3>
                    <div class="metric-value">)" << std::fixed << std::setprecision(1) << (max_memory / 1024.0 / 1024.0) << R"( MB</div>
                    <div class="metric-label">Maximum memory used</div>
                </div>
        </div>
        )";
    }
    
    html << R"(
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Iteration</th>
                    <th>Add Time (ms)</th>
                    <th>Search Time (ms)</th>
                    <th>Add Throughput (v/s)</th>
                    <th>Search Throughput (q/s)</th>
                    <th>Overall Throughput (ops/s)</th>
                    <th>Memory Usage (MB)</th>
                    <th>Recall @k</th>
                    <th>Precision @k</th>
                    <th>F1 Score</th>
                </tr>
        )";
    
    for (size_t i = 0; i < data_.size(); ++i) {
        const auto& m = data_[i];
        html << R"(
                <tr>
                    <td>)" << (i + 1) << R"(</td>
                    <td>)" << std::fixed << std::setprecision(2) << m.add_time_ms << R"(</td>
                    <td>)" << std::fixed << std::setprecision(2) << m.search_time_ms << R"(</td>
                    <td>)" << std::fixed << std::setprecision(0) << m.add_throughput_vps << R"(</td>
                    <td>)" << std::fixed << std::setprecision(0) << m.search_throughput_qps << R"(</td>
                    <td>)" << std::fixed << std::setprecision(0) << m.overall_throughput_ops << R"(</td>
                    <td>)" << std::fixed << std::setprecision(1) << (m.memory_usage_bytes / 1024.0 / 1024.0) << R"(</td>
                    <td>)" << std::fixed << std::setprecision(3) << m.recall_at_k << R"(</td>
                    <td>)" << std::fixed << std::setprecision(3) << m.precision_at_k << R"(</td>
                    <td>)" << std::fixed << std::setprecision(3) << m.f1_score << R"(</td>
                </tr>
        )";
    }
    
    html << R"(
            </table>
        </div>
    </div>
</body>
</html>
)";
    
    std::ofstream file("performance_report.html");
    file << html.str();
    file.close();
}

void PerformanceVisualizer::export_visualization(const std::string& filename) {
    if (output_format_ == "html") {
        generate_html_report();
    } else {
        generate_charts();
    }
}

// BenchmarkReport Implementation
BenchmarkReport::BenchmarkReport(const std::vector<BenchmarkMetrics>& results) 
    : results_(results) {}

void BenchmarkReport::generate_comprehensive_report() {
    std::cout << "Generating comprehensive benchmark report..." << std::endl;
    
    // Calculate statistics
    std::vector<double> add_times, search_times, throughputs;
    for (const auto& m : results_) {
        add_times.push_back(m.add_time_ms);
        search_times.push_back(m.search_time_ms);
        throughputs.push_back(m.overall_throughput_ops);
    }
    
    // Sort for percentiles
    std::sort(search_times.begin(), search_times.end());
    
    std::cout << "Report includes " << results_.size() << " data points" << std::endl;
}

void BenchmarkReport::generate_executive_summary() {
    std::cout << "\n=== EXECUTIVE SUMMARY ===" << std::endl;
    
    if (results_.empty()) return;
    
    // Calculate averages
    double avg_add = 0, avg_search = 0, avg_throughput = 0;
    for (const auto& m : results_) {
        avg_add += m.add_time_ms;
        avg_search += m.search_time_ms;
        avg_throughput += m.overall_throughput_ops;
    }
    
    avg_add /= results_.size();
    avg_search /= results_.size();
    avg_throughput /= results_.size();
    
    std::cout << "Performance Summary:" << std::endl;
    std::cout << "  Average Add Time: " << format_metric(avg_add, "ms") << std::endl;
    std::cout << "  Average Search Time: " << format_metric(avg_search, "ms") << std::endl;
    std::cout << "  Average Throughput: " << format_throughput(avg_throughput) << std::endl;
    
    // Find best and worst performers
    auto best_add = std::min_element(results_.begin(), results_.end(), 
        [](const BenchmarkMetrics& a, const BenchmarkMetrics& b) {
            return a.add_time_ms < b.add_time_ms;
        });
    
    auto worst_search = std::max_element(results_.begin(), results_.end(),
        [](const BenchmarkMetrics& a, const BenchmarkMetrics& b) {
            return a.search_time_ms > b.search_time_ms;
        });
    
    std::cout << "\nExtremes:" << std::endl;
    std::cout << "  Best Add Time: " << format_metric(best_add->add_time_ms, "ms") << std::endl;
    std::cout << "  Worst Search Time: " << format_metric(worst_search->search_time_ms, "ms") << std::endl;
}

void BenchmarkReport::generate_detailed_analysis() {
    std::cout << "\n=== DETAILED ANALYSIS ===" << std::endl;
    
    if (results_.empty()) return;
    
    // Performance consistency analysis
    std::vector<double> search_times;
    for (const auto& m : results_) {
        search_times.push_back(m.search_time_ms);
    }
    
    double mean = std::accumulate(search_times.begin(), search_times.end(), 0.0) / search_times.size();
    double variance = 0.0;
    for (double time : search_times) {
        variance += (time - mean) * (time - mean);
    }
    variance /= search_times.size();
    double std_dev = std::sqrt(variance);
    
    double cv = (std_dev / mean) * 100;  // Coefficient of variation
    
    std::cout << "Performance Consistency:" << std::endl;
    std::cout << "  Mean Search Time: " << format_metric(mean, "ms") << std::endl;
    std::cout << "  Standard Deviation: " << format_metric(std_dev, "ms") << std::endl;
    std::cout << "  Coefficient of Variation: " << format_metric(cv, "%") << std::endl;
    
    // Memory analysis
    double avg_memory = 0;
    for (const auto& m : results_) {
        avg_memory += m.memory_usage_bytes;
    }
    avg_memory /= results_.size();
    
    std::cout << "\nMemory Analysis:" << std::endl;
    std::cout << "  Average Memory Usage: " << format_size(static_cast<size_t>(avg_memory)) << std::endl;
}

void BenchmarkReport::generate_recommendations() {
    std::cout << "\n=== RECOMMENDATIONS ===" << std::endl;
    
    if (results_.empty()) return;
    
    // Performance recommendations
    double avg_search_time = 0;
    for (const auto& m : results_) {
        avg_search_time += m.search_time_ms;
    }
    avg_search_time /= results_.size();
    
    std::cout << "Performance Recommendations:" << std::endl;
    
    if (avg_search_time > 100) {
        std::cout << "  - Consider using GPU for better performance with large datasets" << std::endl;
    }
    
    if (avg_search_time > 50) {
        std::cout << "  - Consider using IVF index for faster approximate search" << std::endl;
    }
    
    if (avg_search_time < 10) {
        std::cout << "  - Excellent performance achieved" << std::endl;
    }
    
    // Memory recommendations
    double avg_memory = 0;
    for (const auto& m : results_) {
        avg_memory += m.memory_usage_bytes;
    }
    avg_memory /= results_.size();
    
    if (avg_memory > 1024 * 1024 * 1024) {  // > 1GB
        std::cout << "  - Consider memory optimization techniques for large datasets" << std::endl;
    }
    
    std::cout << "\nConfiguration Recommendations:" << std::endl;
    std::cout << "  - Use batch sizes between 32-256 for optimal throughput" << std::endl;
    std::cout << "  - Enable C++ optimizations for production workloads" << std::endl;
    std::cout << "  - Monitor memory usage for scaling decisions" << std::endl;
}

void BenchmarkReport::export_report(const std::string& format, const std::string& filename) {
    if (format == "json") {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"benchmark_results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& m = results_[i];
            
            file << "    {\n";
            file << "      \"add_time_ms\": " << m.add_time_ms << ",\n";
            file << "      \"search_time_ms\": " << m.search_time_ms << ",\n";
            file << "      \"add_throughput_vps\": " << m.add_throughput_vps << ",\n";
            file << "      \"search_throughput_qps\": " << m.search_throughput_qps << ",\n";
            file << "      \"overall_throughput_ops\": " << m.overall_throughput_ops << ",\n";
            file << "      \"memory_usage_bytes\": " << m.memory_usage_bytes << ",\n";
            file << "      \"recall_at_k\": " << m.recall_at_k << ",\n";
            file << "      \"precision_at_k\": " << m.precision_at_k << ",\n";
            file << "      \"f1_score\": " << m.f1_score << "\n";
            file << "    }";
            
            if (i < results_.size() - 1) {
                file << ",";
            }
        }
        
        file << "  ]\n";
        file << "}\n";
        file.close();
    }
}

std::string BenchmarkReport::format_metric(double value, const std::string& unit) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value << " " << unit;
    return ss.str();
}

std::string BenchmarkReport::format_size(size_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / 1024 / 1024) + " MB";
    } else {
        return std::to_string(bytes / 1024 / 1024 / 1024) + " GB";
    }
}

std::string BenchmarkReport::format_throughput(double ops_per_sec) {
    if (ops_per_sec < 1000) {
        return std::to_string(static_cast<int>(ops_per_sec)) + " ops/s";
    } else if (ops_per_sec < 1000 * 1000) {
        return std::to_string(static_cast<int>(ops_per_sec / 1000)) + " K ops/s";
    } else {
        return std::to_string(static_cast<int>(ops_per_sec / 1000 / 1000)) + " M ops/s";
    }
}

} // namespace hybridvectordb
