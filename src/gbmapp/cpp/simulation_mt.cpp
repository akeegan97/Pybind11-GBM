// gbm_mt.cpp
#include "simulation_common.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <stdexcept>

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

static inline void WorkerLogSpace(
    int numPaths,
    int steps,
    double logStartPrice,
    double partialComputation,
    double volStep, // normalizedStd * sqrtDeltaT
    PaddedDouble& outSumFinalPrices,
    std::vector<std::vector<double>>* outDisplayPaths,
    int maxDisplayPaths
) {
    double sum = 0.0;
    double comp = 0.0; // Kahan comp

    // Seed RNG with random_device for true randomness
    // For reproducible results, could use a seed based on thread ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> nd(0.0, 1.0);

    for (int i = 0; i < numPaths; ++i) {
        double lp = logStartPrice;

        const bool collect = (outDisplayPaths != nullptr) &&
                             (static_cast<int>(outDisplayPaths->size()) < maxDisplayPaths);

        std::vector<double> path;
        if (collect) {
            path.reserve(steps);
            path.push_back(std::exp(logStartPrice));
        }

        for (int j = 1; j < steps; ++j) {
            lp += partialComputation + volStep * nd(gen);
            if (collect) path.push_back(std::exp(lp));
        }

        KahanAdd(sum, comp, std::exp(lp));
        if (collect) outDisplayPaths->push_back(std::move(path));
    }

    outSumFinalPrices.v = sum;
}

} // namespace

SimulationResult SimulateGBMMultiThreaded(
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

    // Generate display paths first (separate from parallel computation)
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

        threads.emplace_back(
            WorkerLogSpace,
            threadPaths,
            steps,
            logStart,
            partialComp,
            volStep,
            std::ref(sums[t]),
            nullptr,  // No display path collection in workers
            0
        );
    }

    for (auto& th : threads) th.join();

    double totalSum = 0.0;
    for (const auto& s : sums) totalSum += s.v;

    const double avg = totalSum / static_cast<double>(paths);
    return {displayPaths, avg};
}

} // namespace gbm
