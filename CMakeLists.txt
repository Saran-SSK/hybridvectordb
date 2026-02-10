cmake_minimum_required(3.15)
project(HybridVectorDB_cpp VERSION 0.4.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find FAISS
find_path(FAISS_INCLUDE_DIR include/faiss/VectorIndex.h
    PATHS
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    /usr/include
    /usr/local/include
    DOC "FAISS include directory"
)

if(FAISS_INCLUDE_DIR)
    set(FAISS_FOUND TRUE)
    message(STATUS "Found FAISS include: ${FAISS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "FAISS headers not found")
endif()

# Find FAISS library
find_library(FAISS_LIBRARY faiss
    PATHS
    ${FAISS_INCLUDE_DIR}
    /usr/lib
    /usr/local/lib
    DOC "FAISS library"
)

if(FAISS_LIBRARY)
    message(STATUS "Found FAISS library: ${FAISS_LIBRARY}")
else()
    message(FATAL_ERROR "FAISS library not found")
endif()

# Python includes
pybind11_add_module(HybridVectorDB_cpp
    VERSION ${PROJECT_VERSION}
    CLASS HybridVectorDB
    MODULE_NAME HybridVectorDB._cpp
    SHARED
    EXCLUDE_FROM_ALL
    SYSTEM
    NO_EXTRAS
    THIRDPARTY
)

# Source files
target_sources(HybridVectorDB_cpp
    PRIVATE
    src/cpp/vector_database.cpp
    src/cpp/index_manager.cpp
    src/cpp/memory_manager.cpp
    src/cpp/search_engine.cpp
    src/cpp/optimizations.cpp
    src/cpp/utils.cpp
)

# Include directories
target_include_directories(HybridVectorDB_cpp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${FAISS_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(HybridVectorDB_cpp
    PRIVATE
    ${FAISS_LIBRARY}
    ${pybind11_LIBRARIES}
)

# Compiler flags
target_compile_definitions(HybridVectorDB_cpp
    PRIVATE
    HYBRIDVECTORDB_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    HYBRIDVECTORDB_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    HYBRIDVECTORDB_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

target_compile_options(HybridVectorDB_cpp
    PRIVATE
    $<$<COMPILE_LANGUAGE>:CXX>
    $<$<CXX_COMPILER_ID>:GNU>:-Wall -Wextra -O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2 /arch:AVX2>
)

# Linker flags
target_link_options(HybridVectorDB_cpp
    PRIVATE
    $<$<PLATFORM_ID:Windows>:/OPT:REF /LTCG>
    $<$<PLATFORM_ID:Linux>:-Wl,--as-needed>
)

# Installation
install(TARGETS HybridVectorDB_cpp
    COMPONENT HybridVectorDB
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/hybridvectordb/HybridVectorDB.py
    COMPONENT HybridVectorDB
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
)

# Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "HybridVectorDB C++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  FAISS Found: ${FAISS_FOUND}")
message(STATUS "  pybind11 Found: ${pybind11_FOUND}")
message(STATUS "")
