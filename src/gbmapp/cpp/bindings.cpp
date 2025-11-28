// Python bindings for GBM simulation module
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simulation_common.h"

namespace py = pybind11;

// Forward declarations of simulation functions
namespace gbm {
    SimulationResult SimulateGBMScalar(double, double, double, double, int, int);
    SimulationResult SimulateGBMMultiThreaded(double, double, double, double, int, int);
    SimulationResult SimulateGBMIntrinsicMT(double, double, double, double, int, int);
}

PYBIND11_MODULE(simulation, m) {
    m.doc() = "High-performance GBM simulation module with multiple engine implementations";
    
    // System capabilities structure
    py::class_<gbm::SystemCapabilities>(m, "SystemCapabilities")
        .def_readonly("has_avx2", &gbm::SystemCapabilities::has_avx2,
            "Whether AVX2 instructions are supported")
        .def_readonly("has_avx512", &gbm::SystemCapabilities::has_avx512,
            "Whether AVX-512 instructions are supported")
        .def_readonly("num_threads", &gbm::SystemCapabilities::num_threads,
            "Number of hardware threads available")
        .def_readonly("cache_line_size", &gbm::SystemCapabilities::cache_line_size,
            "CPU cache line size in bytes");
    
    // System capability detection
    m.def("GetSystemCapabilities", &gbm::GetSystemCapabilities,
        "Get system hardware capabilities for optimal engine selection");
    
    // Scalar (single-threaded) implementation
    m.def("SimulateGBMScalar", &gbm::SimulateGBMScalar,
        "Simulate GBM paths using scalar (single-threaded) implementation",
        py::arg("starting_price"),
        py::arg("normalized_mu"),
        py::arg("normalized_var"),
        py::arg("normalized_std"),
        py::arg("steps"),
        py::arg("paths"),
        R"pbdoc(
            Simulate Geometric Brownian Motion paths using scalar implementation.
            
            This is a single-threaded implementation useful for small simulations
            or as a baseline for performance comparison.
            
            Parameters:
                starting_price: Initial stock price (must be > 0)
                normalized_mu: Drift coefficient (normalized for time period)
                normalized_var: Variance coefficient (normalized for time period)
                normalized_std: Standard deviation (volatility coefficient)
                steps: Number of time steps in each path (must be > 0)
                paths: Number of simulation paths (must be > 0)
            
            Returns:
                Tuple of (display_paths, average_final_price) where display_paths
                contains up to 50 complete price paths for visualization
        )pbdoc");
    
    // Multi-threaded implementation
    m.def("SimulateGBMMultiThreaded", &gbm::SimulateGBMMultiThreaded,
        "Simulate GBM paths using multi-threaded implementation",
        py::arg("starting_price"),
        py::arg("normalized_mu"),
        py::arg("normalized_var"),
        py::arg("normalized_std"),
        py::arg("steps"),
        py::arg("paths"),
        R"pbdoc(
            Simulate Geometric Brownian Motion paths using multi-threaded implementation.
            
            Automatically distributes work across all available CPU cores for
            improved performance on multi-core systems.
            
            Parameters:
                starting_price: Initial stock price (must be > 0)
                normalized_mu: Drift coefficient (normalized for time period)
                normalized_var: Variance coefficient (normalized for time period)
                normalized_std: Standard deviation (volatility coefficient)
                steps: Number of time steps in each path (must be > 0)
                paths: Number of simulation paths (must be > 0)
            
            Returns:
                Tuple of (display_paths, average_final_price) where display_paths
                contains up to 50 complete price paths for visualization
        )pbdoc");
    
    // SIMD-optimized multi-threaded implementation
    m.def("SimulateGBMIntrinsicMT", &gbm::SimulateGBMIntrinsicMT,
        "Simulate GBM paths using SIMD-optimized multi-threaded implementation",
        py::arg("starting_price"),
        py::arg("normalized_mu"),
        py::arg("normalized_var"),
        py::arg("normalized_std"),
        py::arg("steps"),
        py::arg("paths"),
        R"pbdoc(
            Simulate Geometric Brownian Motion paths using SIMD-optimized implementation.
            
            Uses AVX2 intrinsic instructions to process 4 paths simultaneously
            on each thread, providing maximum performance on modern CPUs.
            
            Requires AVX2 CPU support (check with GetSystemCapabilities()).
            
            Parameters:
                starting_price: Initial stock price (must be > 0)
                normalized_mu: Drift coefficient (normalized for time period)
                normalized_var: Variance coefficient (normalized for time period)
                normalized_std: Standard deviation (volatility coefficient)
                steps: Number of time steps in each path (must be > 0)
                paths: Number of simulation paths (must be > 0)
            
            Returns:
                Tuple of (display_paths, average_final_price) where display_paths
                contains up to 50 complete price paths for visualization
        )pbdoc");
    
    // Alias for backward compatibility
    m.def("SimulateGBM", &gbm::SimulateGBMScalar,
        "Alias for SimulateGBMScalar (backward compatibility)",
        py::arg("starting_price"),
        py::arg("normalized_mu"),
        py::arg("normalized_var"),
        py::arg("normalized_std"),
        py::arg("steps"),
        py::arg("paths"));
    
    // Parameter validation function
    m.def("ValidateParameters", &gbm::ValidateParameters,
        "Validate simulation parameters",
        py::arg("starting_price"),
        py::arg("normalized_mu"),
        py::arg("normalized_var"),
        py::arg("normalized_std"),
        py::arg("steps"),
        py::arg("paths"),
        R"pbdoc(
            Validate simulation parameters before running simulation.
            
            Raises:
                ValueError: If any parameter is invalid
            
            Returns:
                True if all parameters are valid
        )pbdoc");
}
