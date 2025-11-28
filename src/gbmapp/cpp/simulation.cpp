// Common utilities and system capability detection for GBM simulations
#include "simulation_common.h"
#include <thread>
#include <cmath>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

namespace gbm {

SystemCapabilities GetSystemCapabilities() {
    SystemCapabilities caps;
    
    // Get number of hardware threads
    caps.num_threads = std::thread::hardware_concurrency();
    if (caps.num_threads == 0) {
        caps.num_threads = 1;  // Fallback
    }
    
    // Detect AVX2 and AVX512 support
    caps.has_avx2 = false;
    caps.has_avx512 = false;
    
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        caps.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
        caps.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
    }
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.has_avx2 = (ebx & (1 << 5)) != 0;
        caps.has_avx512 = (ebx & (1 << 16)) != 0;
    }
#endif
    
    // Cache line size (typically 64 bytes)
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    caps.cache_line_size = sysInfo.dwProcessorType;  // Approximation
#else
    long cache_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    caps.cache_line_size = (cache_size > 0) ? static_cast<uint32_t>(cache_size) : 64;
#endif
    
    if (caps.cache_line_size == 0) {
        caps.cache_line_size = 64;  // Default fallback
    }
    
    return caps;
}

bool ValidateParameters(double startingPrice, double normalizedMu,
                       double normalizedVar, double normalizedStd,
                       int steps, int paths) {
    if (startingPrice <= 0) {
        throw std::invalid_argument("Starting price must be positive");
    }
    if (steps <= 0) {
        throw std::invalid_argument("Steps must be positive");
    }
    if (paths <= 0) {
        throw std::invalid_argument("Paths must be positive");
    }
    if (normalizedVar < 0) {
        throw std::invalid_argument("Variance cannot be negative");
    }
    if (normalizedStd < 0) {
        throw std::invalid_argument("Standard deviation cannot be negative");
    }
    if (!std::isfinite(startingPrice) || !std::isfinite(normalizedMu) ||
        !std::isfinite(normalizedVar) || !std::isfinite(normalizedStd)) {
        throw std::invalid_argument("Parameters must be finite");
    }
    return true;
}

} // namespace gbm
