from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# C++ extension
ext_modules = [
    Pybind11Extension(
        "hybridvectordb._cpp",
        sources=[
            "src/cpp/bindings.cpp",
            "src/cpp/vector_database.cpp",
            "src/cpp/index_manager.cpp",
            "src/cpp/memory_manager.cpp",
            "src/cpp/search_engine.cpp",
            "src/cpp/optimizations.cpp",
            "src/cpp/utils.cpp",
            "src/cpp/benchmark.cpp",
            "src/cpp/visualization.cpp",
            "src/cpp/testing.cpp"
        ],
        include_dirs=[
            "src/cpp",
            get_cmake_dir(),
        ],
        cxx_std=17,
        define_macros=[("HYBRIDVECTORDB_VERSION_MAJOR", "0"),
                     ("HYBRIDVECTORDB_VERSION_MINOR", "5"),
                     ("HYBRIDVECTORDB_VERSION_PATCH", "0")],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="hybridvectordb",
    version="0.5.0",
    description="Hybrid CPU/GPU vector database with intelligent query routing and comprehensive benchmarking",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "faiss-cpu>=1.7.4",
        "numpy>=1.21.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "pynvml>=11.0.0",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "cpp": ["pybind11>=2.10.0", "cmake>=3.15.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "mypy>=1.0.0", "gtest>=1.11.0"],
        "benchmark": ["matplotlib>=3.5.0", "seaborn>=0.11.0", "jupyter>=1.0.0"],
        "production": ["gunicorn>=21.2.0", "flask>=2.3.2", "prometheus-client>=0.16.0", 
                      "psutil>=5.9.5", "uvicorn>=0.21.0", "fastapi>=0.95.0"],
        "all": ["faiss-gpu>=1.7.4", "pybind11>=2.10.0", "cmake>=3.15.0", 
                "pytest>=7.0.0", "black>=22.0.0", "mypy>=1.0.0", "gtest>=1.11.0",
                "matplotlib>=3.5.0", "seaborn>=0.11.0", "jupyter>=1.0.0",
                "gunicorn>=21.2.0", "flask>=2.3.2", "prometheus-client>=0.16.0", 
                "psutil>=5.9.5", "uvicorn>=0.21.0", "fastapi>=0.95.0"]
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
