// gbm_simd_mt.cpp
#include "simulation_common.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <stdexcept>

#include <immintrin.h>

namespace gbm {

static inline void RequireValidParams(
    double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,
    int steps, int paths
) {
    if (!ValidateParameters(startingPrice, normalizedMu, normalizedVar, normalizedStd, steps, paths)) {
        throw std::invalid_argument("Invalid GBM parameters");
    }
}

namespace {

struct alignas(64) PaddedDouble { double v = 0.0; };

double CalculateSIMDPathsSum_LogSpace(
    int numPaths,
    int steps,
    double logStartPrice,
    double partialComputation,
    double volStep
) {
    double sum = 0.0;
    double comp = 0.0;

    // Seed RNG with random_device for true randomness
    // For reproducible results, could use a seed based on thread ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> nd(0.0, 1.0);

#if defined(__AVX2__) && defined(__FMA__)
    const int simdPaths = (numPaths / 4) * 4;

    const __m256d v_partial = _mm256_set1_pd(partialComputation);
    const __m256d v_a = _mm256_set1_pd(volStep);
    const __m256d v_logStart = _mm256_set1_pd(logStartPrice);

    alignas(32) double tmp[4];

    for (int i = 0; i < simdPaths; i += 4) {
        __m256d v_lp = v_logStart;

        for (int j = 1; j < steps; ++j) {
            // Biggest bottleneck for small steps: generating 4 normals (scalar).
            const __m256d v_z = _mm256_set_pd(nd(gen), nd(gen), nd(gen), nd(gen));
            const __m256d v_inc = _mm256_fmadd_pd(v_a, v_z, v_partial);
            v_lp = _mm256_add_pd(v_lp, v_inc);
        }

        _mm256_store_pd(tmp, v_lp);
        KahanAdd(sum, comp, std::exp(tmp[0]));
        KahanAdd(sum, comp, std::exp(tmp[1]));
        KahanAdd(sum, comp, std::exp(tmp[2]));
        KahanAdd(sum, comp, std::exp(tmp[3]));
    }

    for (int i = simdPaths; i < numPaths; ++i) {
        double lp = logStartPrice;
        for (int j = 1; j < steps; ++j) lp += partialComputation + volStep * nd(gen);
        KahanAdd(sum, comp, std::exp(lp));
    }

    return sum;
#else
    // Fallback if compiled without AVX2+FMA flags.
    for (int i = 0; i < numPaths; ++i) {
        double lp = logStartPrice;
        for (int j = 1; j < steps; ++j) lp += partialComputation + volStep * nd(gen);
        KahanAdd(sum, comp, std::exp(lp));
    }
    return sum;
#endif
}

static inline void SIMDWorker(
    int numPaths,
    int steps,
    double logStartPrice,
    double partialComputation,
    double volStep,
    PaddedDouble& outSum
) {
    outSum.v = CalculateSIMDPathsSum_LogSpace(numPaths, steps, logStartPrice, partialComputation, volStep);
}

} // namespace

SimulationResult SimulateGBMIntrinsicMT(
    double startingPrice,
    double normalizedMu,
    double normalizedVar,
    double normalizedStd,
    int steps,
    int paths,
    int displayPathsRequested
) {
    RequireValidParams(startingPrice, normalizedMu, normalizedVar, normalizedStd, steps, paths);

    displayPathsRequested = std::max(0, std::min(displayPathsRequested, paths));

    const double deltaT = 1.0 / steps;
    const double partialComp = (normalizedMu - 0.5 * normalizedVar) * deltaT;
    const double volStep = normalizedStd * std::sqrt(deltaT);
    const double logStart = std::log(startingPrice);

    // Display paths (scalar, log-space, cheap)
    std::vector<std::vector<double>> displayPaths;
    displayPaths.reserve(static_cast<size_t>(displayPathsRequested));
    if (displayPathsRequested > 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> nd(0.0, 1.0);

        for (int i = 0; i < displayPathsRequested; ++i) {
            std::vector<double> path;
            path.reserve(steps);
            path.push_back(startingPrice);

            double lp = logStart;
            for (int j = 1; j < steps; ++j) {
                lp += partialComp + volStep * nd(gen);
                path.push_back(std::exp(lp));
            }
            displayPaths.push_back(std::move(path));
        }
    }

    auto caps = GetSystemCapabilities();
    unsigned int numThreads = std::max(1u, caps.num_threads);
    numThreads = std::min<unsigned int>(numThreads, static_cast<unsigned int>(paths));

    const int pathsPerThread = paths / static_cast<int>(numThreads);
    const int rem = paths % static_cast<int>(numThreads);

    std::vector<PaddedDouble> sums(numThreads);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; ++t) {
        const int threadPaths = pathsPerThread + (t < static_cast<unsigned>(rem) ? 1 : 0);
        threads.emplace_back(SIMDWorker, threadPaths, steps, logStart, partialComp, volStep, std::ref(sums[t]));
    }

    for (auto& th : threads) th.join();

    double totalSum = 0.0;
    for (const auto& s : sums) totalSum += s.v;

    const double avg = totalSum / static_cast<double>(paths);
    return {displayPaths, avg};
}

} // namespace gbm
