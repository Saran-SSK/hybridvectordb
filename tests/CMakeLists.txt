cmake_minimum_required(3.15)

# Find required packages
find_package(GTest REQUIRED)

# Test executable
add_executable(test_cpp 
    test_cpp.cpp
    ../src/cpp/vector_database.cpp
    ../src/cpp/index_manager.cpp
    ../src/cpp/memory_manager.cpp
    ../src/cpp/search_engine.cpp
    ../src/cpp/optimizations.cpp
    ../src/cpp/utils.cpp
)

# Include directories
target_include_directories(test_cpp
    PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ../src/cpp
)

# Link libraries
target_link_libraries(test_cpp
    PRIVATE
    ${GTEST_LIBRARIES}
    pthread
)

# Link with main library
target_link_libraries(test_cpp
    PRIVATE
    HybridVectorDB_cpp
)

# Add test to CTest
add_test(NAME HybridVectorDB_CPP_Tests COMMAND test_cpp)

# Enable testing
enable_testing()
