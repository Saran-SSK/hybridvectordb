#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include "vector_database.h"

namespace py = pybind11;

PYBIND11_MODULE(HybridVectorDB_cpp, m) {
    m.doc() = "HybridVectorDB C++ bindings with zero-copy optimizations";
    
    // Bind Config struct
    py::class_<hybridvectordb::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dimension", &hybridvectordb::Config::dimension)
        .def_readwrite("index_type", &hybridvectordb::Config::index_type)
        .def_readwrite("metric_type", &hybridvectordb::Config::metric_type)
        .def_readwrite("use_gpu", &hybridvectordb::Config::use_gpu)
        .def_readwrite("batch_threshold", &hybridvectordb::Config::batch_threshold)
        .def_readwrite("k_threshold", &hybridvectordb::Config::k_threshold)
        .def_readwrite("dataset_threshold", &hybridvectordb::Config::dataset_threshold)
        .def_readwrite("gpu_util_limit", &hybridvectordb::Config::gpu_util_limit);
    
    // Bind VectorData struct
    py::class_<hybridvectordb::VectorData>(m, "VectorData")
        .def(py::init<const std::string&, const std::vector<float>&>())
        .def_readwrite("id", &hybridvectordb::VectorData::id)
        .def_readwrite("embedding", &hybridvectordb::VectorData::embedding)
        .def_readwrite("metadata", &hybridvectordb::VectorData::metadata);
    
    // Bind SearchResult struct
    py::class_<hybridvectordb::SearchResult>(m, "SearchResult")
        .def(py::init<const std::string&, float>())
        .def_readwrite("id", &hybridvectordb::SearchResult::id)
        .def_readwrite("distance", &hybridvectordb::SearchResult::distance)
        .def_readwrite("metadata", &hybridvectordb::SearchResult::metadata);
    
    // Bind SearchResponse struct
    py::class_<hybridvectordb::SearchResponse>(m, "SearchResponse")
        .def(py::init<const std::string&, 
                     const std::vector<hybridvectordb::SearchResult>&,
                     double,
                     const std::string&>())
        .def_readwrite("query_id", &hybridvectordb::SearchResponse::query_id)
        .def_readwrite("results", &hybridvectordb::SearchResponse::results)
        .def_readwrite("total_results", &hybridvectordb::SearchResponse::total_results)
        .def_readwrite("search_time_ms", &hybridvectordb::SearchResponse::search_time_ms)
        .def_readwrite("index_used", &hybridvectordb::SearchResponse::index_used);
    
    // Bind PerformanceMetrics struct
    py::class_<hybridvectordb::PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_property("total_queries", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.total_queries.load(); })
        .def_property("cpu_queries", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.cpu_queries.load(); })
        .def_property("gpu_queries", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.gpu_queries.load(); })
        .def_property("total_search_time_ms", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.total_search_time_ms.load(); })
        .def_property("avg_cpu_time_ms", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.cpu_time_ms.load(); })
        .def_property("avg_gpu_time_ms", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.gpu_time_ms.load(); })
        .def_property("cpu_success_rate", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.cpu_success_rate.load(); })
        .def_property("gpu_success_rate", 
                    [](const hybridvectordb::PerformanceMetrics& m) { return m.gpu_success_rate.load(); })
        .def("get_speedup", &hybridvectordb::PerformanceMetrics::get_speedup);
    
    // Bind main HybridVectorDB class
    py::class_<hybridvectordb::HybridVectorDB>(m, "HybridVectorDB")
        .def(py::init<const hybridvectordb::Config&>())
        .def("add_vectors", 
              &hybridvectordb::HybridVectorDB::add_vectors,
              py::arg("vectors"), 
              py::arg("count"),
              "Add vectors to database")
        .def("search_vectors", 
              &hybridvectordb::HybridVectorDB::search_vectors,
              py::arg("query_vectors"),
              py::arg("query_count"), 
              py::arg("k"),
              py::arg("use_gpu") = false,
              "Search for similar vectors")
        .def("get_metrics", 
              &hybridvectordb::HybridVectorDB::get_metrics,
              "Get performance metrics")
        .def("reset_metrics", 
              &hybridvectordb::HybridVectorDB::reset_metrics,
              "Reset performance metrics")
        .def("get_stats", 
              &hybridvectordb::HybridVectorDB::get_stats,
              "Get database statistics")
        .def("configure", 
              &hybridvectordb::HybridVectorDB::configure,
              "Configure database")
        .def("optimize_performance", 
              &hybridvectordb::HybridVectorDB::optimize_performance,
              py::arg("operation"),
              "Optimize performance")
        .def("get_vectors_zero_copy", 
              &hybridvectordb::HybridVectorDB::get_vectors_zero_copy,
              py::arg("count"),
              py::return_value_policy::reference_internal,
              "Get vectors with zero-copy access")
        .def("get_memory_usage", 
              &hybridvectordb::HybridVectorDB::get_memory_usage,
              "Get memory usage information")
        .def("benchmark", 
              &hybridvectordb::HybridVectorDB::benchmark,
              py::arg("num_vectors"),
              py::arg("num_queries"),
              py::arg("k"),
              "Benchmark performance");
    
    // Bind factory function
    m.def("create_vector_database", 
           &hybridvectordb::create_vector_database,
           py::arg("config"),
           "Create HybridVectorDB instance");
    
    // Bind optimization utilities
    py::module_ optimization = m.def_submodule("optimization", "Performance optimization utilities");
    
    optimization.def("optimize_batch_size", 
                   &hybridvectordb::optimization::optimize_batch_size,
                   py::arg("operation"),
                   py::arg("test_sizes"),
                   "Optimize batch size for given operation");
    
    optimization.def("optimize_memory_layout", 
                   &hybridvectordb::optimization::optimize_memory_layout,
                   py::arg("data"), 
                   py::arg("count"), 
                   py::arg("dimension"),
                   "Optimize memory layout");
    
    optimization.def("apply_simd_optimizations", 
                   &hybridvectordb::optimization::apply_simd_optimizations,
                   py::arg("data"), 
                   py::arg("count"), 
                   py::arg("dimension"),
                   "Apply SIMD optimizations");
    
    // Bind error handling
    py::register_exception<hybridvectordb::HybridVectorDBError>(m, "HybridVectorDBError");
    
    // Bind numpy array helpers
    m.def("add_vectors_numpy", 
           [](hybridvectordb::HybridVectorDB& db, py::array_t<float> vectors, py::list ids, py::list metadata) {
               // Convert numpy array to C++ format
               py::buffer_info buf = vectors.request();
               float* ptr = static_cast<float*>(buf.ptr);
               size_t count = buf.shape[0];
               size_t dim = buf.shape[1];
               
               std::vector<hybridvectordb::VectorData> vector_data;
               vector_data.reserve(count);
               
               for (size_t i = 0; i < count; ++i) {
                   std::string id = ids[i].cast<std::string>();
                   std::vector<float> embedding(ptr + i * dim, ptr + (i + 1) * dim);
                   
                   std::unordered_map<std::string, std::string> meta;
                   if (!metadata.empty()) {
                       py::dict meta_dict = metadata[i].cast<py::dict>();
                       for (auto item : meta_dict) {
                           meta[item.first.cast<std::string>()] = item.second.cast<std::string>();
                       }
                   }
                   
                   vector_data.emplace_back(id, embedding);
                   vector_data.back().metadata = meta;
               }
               
               return db.add_vectors(vector_data.data(), vector_data.size());
           },
           py::arg("db"), 
           py::arg("vectors"), 
           py::arg("ids") = py::list(),
           py::arg("metadata") = py::list(),
           "Add vectors from numpy array");
    
    m.def("search_vectors_numpy", 
           [](hybridvectordb::HybridVectorDB& db, py::array_t<float> queries, size_t k, bool use_gpu) {
               py::buffer_info buf = queries.request();
               float* ptr = static_cast<float*>(buf.ptr);
               size_t query_count = buf.shape[0];
               
               return db.search_vectors(ptr, query_count, k, use_gpu);
           },
           py::arg("db"), 
           py::arg("queries"), 
           py::arg("k"), 
           py::arg("use_gpu") = false,
           "Search vectors from numpy array");
    
    // Version information
    m.attr("__version__") = "0.4.0";
    m.attr("__cpp_version__") = HYBRIDVECTORDB_VERSION_MAJOR "." HYBRIDVECTORDB_VERSION_MINOR "." HYBRIDVECTORDB_VERSION_PATCH;
}
