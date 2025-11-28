// Common header for GBM simulation implementations
#pragma once

#include <vector>
#include <utility>
#include <cstdint>

namespace gbm {

// Simulation result type: (display_paths, average_final_price)
using SimulationResult = std::pair<std::vector<std::vector<double>>, double>;

// System capabilities structure
struct SystemCapabilities {
    bool has_avx2;
    bool has_avx512;
    uint32_t num_threads;
    uint32_t cache_line_size;
};

// Get system capabilities
SystemCapabilities GetSystemCapabilities();

// Validate simulation parameters
bool ValidateParameters(double startingPrice, double normalizedMu, 
                       double normalizedVar, double normalizedStd,
                       int steps, int paths);

} // namespace gbm
