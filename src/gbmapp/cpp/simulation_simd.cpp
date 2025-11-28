// SIMD-optimized multi-threaded GBM simulation implementation
#include "simulation_common.h"
#include <random>
#include <cmath>
#include <thread>
#include <algorithm>
#include <immintrin.h>

namespace gbm {

namespace {

// SIMD custom function for computing exponential function fitted to range (-0.2, 0.2)
// Error compared to std::exp() over same range ~ (7.5E-09:3.1E-10)
__m256d exp_approx(__m256d x) {
    // Coefficients found using numpy and scipy.optimize for the range
    __m256d c0 = _mm256_set1_pd(1.0);
    __m256d c1 = _mm256_set1_pd(1.0);
    __m256d c2 = _mm256_set1_pd(0.49999898);
    __m256d c3 = _mm256_set1_pd(0.16666646);
    __m256d c4 = _mm256_set1_pd(0.04174285);
    __m256d c5 = _mm256_set1_pd(0.00834562);

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x3 = _mm256_mul_pd(x2, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);
    __m256d x5 = _mm256_mul_pd(x3, x2);

    __m256d result = _mm256_add_pd(c0, _mm256_mul_pd(c1, x));
    result = _mm256_add_pd(result, _mm256_mul_pd(c2, x2));
    result = _mm256_add_pd(result, _mm256_mul_pd(c3, x3));
    result = _mm256_add_pd(result, _mm256_mul_pd(c4, x4));
    result = _mm256_add_pd(result, _mm256_mul_pd(c5, x5));

    return result;
}

double CalculateSIMDPaths(
    int numPaths,
    int steps,
    double startingPrice,
    double partialComputation,
    double normalizedStd,
    double sqrtDeltaT
) {
    double sumFinalPrices = 0.0;
    alignas(32) double finalPrices[4];

    // Pre-compute SIMD constants
    __m256d _normalStdVec = _mm256_set1_pd(normalizedStd);
    __m256d _partialCompVec = _mm256_set1_pd(partialComputation);
    __m256d _sqrtDTVec = _mm256_set1_pd(sqrtDeltaT);

    // Thread-local random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    // Process 4 paths at a time using AVX2
    int simdPaths = (numPaths / 4) * 4;
    
    for (int i = 0; i < simdPaths; i += 4) {
        // Initialize prices
        alignas(32) double prices[4];
        for (int k = 0; k < 4; ++k) {
            prices[k] = startingPrice;
        }
        __m256d _prices = _mm256_load_pd(prices);

        // Simulate steps for 4 paths simultaneously
        for (int j = 1; j < steps; ++j) {
            // Generate random numbers
            alignas(32) double ranNums[4];
            for (int k = 0; k < 4; ++k) {
                ranNums[k] = d(gen);
            }
            __m256d _normalDistrValues = _mm256_loadu_pd(ranNums);

            // Compute: exp(partialComputation + normalizedStd * sqrtDeltaT * noise)
            __m256d _a = _mm256_mul_pd(_normalStdVec, _sqrtDTVec);
            __m256d _c = _mm256_fmadd_pd(_a, _normalDistrValues, _partialCompVec);
            __m256d _d = exp_approx(_c);
            _prices = _mm256_mul_pd(_prices, _d);
        }

        // Store final prices and accumulate
        _mm256_storeu_pd(finalPrices, _prices);
        for (int k = 0; k < 4; ++k) {
            sumFinalPrices += finalPrices[k];
        }
    }

    // Handle remaining paths (if numPaths not divisible by 4)
    for (int i = simdPaths; i < numPaths; ++i) {
        double price = startingPrice;
        for (int j = 1; j < steps; ++j) {
            double noise = d(gen);
            price *= std::exp(partialComputation + normalizedStd * sqrtDeltaT * noise);
        }
        sumFinalPrices += price;
    }

    return sumFinalPrices / numPaths;
}

void SIMDWorker(
    int numPaths,
    int steps,
    double startingPrice,
    double partialComputation,
    double normalizedStd,
    double sqrtDeltaT,
    std::vector<double>& averagePrices,
    int threadIndex
) {
    double avg = CalculateSIMDPaths(
        numPaths, steps, startingPrice,
        partialComputation, normalizedStd, sqrtDeltaT
    );
    averagePrices[threadIndex] = avg;
}

} // anonymous namespace

SimulationResult SimulateGBMIntrinsicMT(
    double startingPrice,
    double normalizedMu,
    double normalizedVar,
    double normalizedStd,
    int steps,
    int paths
) {
    // Validate parameters
    ValidateParameters(startingPrice, normalizedMu, normalizedVar, normalizedStd, steps, paths);
    
    // Pre-compute constants
    double deltaT = 1.0 / steps;
    double partialComputation = (normalizedMu - 0.5 * normalizedVar) * deltaT;
    double sqrtDeltaT = std::sqrt(deltaT);

    // Generate display paths (scalar implementation for first 50 paths)
    std::vector<std::vector<double>> displayPaths;
    const int displayPathsCount = std::min(50, paths);
    displayPaths.reserve(displayPathsCount);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for (int i = 0; i < displayPathsCount; ++i) {
        std::vector<double> path;
        path.reserve(steps);
        path.push_back(startingPrice);
        
        double price = startingPrice;
        for (int j = 1; j < steps; ++j) {
            double noise = d(gen);
            price *= std::exp(partialComputation + normalizedStd * sqrtDeltaT * noise);
            path.push_back(price);
        }
        displayPaths.push_back(std::move(path));
    }

    // Determine thread count
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    // Distribute work across threads
    int pathsPerThread = paths / numThreads;
    int remainingPaths = paths % numThreads;

    std::vector<double> averagePrices(numThreads);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    // Launch SIMD worker threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        int threadPaths = pathsPerThread + (i < static_cast<unsigned>(remainingPaths) ? 1 : 0);
        
        threads.emplace_back(
            SIMDWorker,
            threadPaths,
            steps,
            startingPrice,
            partialComputation,
            normalizedStd,
            sqrtDeltaT,
            std::ref(averagePrices),
            i
        );
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Calculate overall average
    double totalAveragePrice = 0.0;
    for (double price : averagePrices) {
        totalAveragePrice += price;
    }
    totalAveragePrice /= numThreads;

    return {displayPaths, totalAveragePrice};
}

} // namespace gbm
