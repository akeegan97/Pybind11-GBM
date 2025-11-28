from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "gbmapp.native.simulation",  # Module path: gbmapp.native.simulation
        [
            "src/gbmapp/cpp/bindings.cpp",       # Python bindings
            "src/gbmapp/cpp/simulation.cpp",     # System capabilities & validation
            "src/gbmapp/cpp/simulation_scalar.cpp",  # Single-threaded implementation
            "src/gbmapp/cpp/simulation_mt.cpp",     # Multi-threaded implementation
            "src/gbmapp/cpp/simulation_simd.cpp",   # SIMD-optimized implementation
        ],
        include_dirs=["src/gbmapp/cpp"],  # For simulation_common.h
        cxx_std=17,
        extra_compile_args=[
            "-O3",           # Maximum optimization
            "-mavx2",        # Enable AVX2 instructions
            "-mfma",         # Enable FMA (fused multiply-add)
            "-march=native", # Optimize for current CPU
            "-fopenmp",      # Enable OpenMP for threading (optional)
        ],
        extra_link_args=[
            "-fopenmp",      # Link OpenMP library (optional)
        ],
    ),
]

setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
