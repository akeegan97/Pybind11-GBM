// Multi-threaded GBM simulation implementation
#include "simulation_common.h"
#include <random>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>
#include <mutex>

namespace gbm {

namespace {

void AddToAtomic(std::atomic<double>& atomicValue, double valueToAdd) {
    double currentValue = atomicValue.load();
    while (!atomicValue.compare_exchange_weak(currentValue, currentValue + valueToAdd));
}

void SimulatePathsWorker(
    int numPaths,
    int steps,
    double startingPrice,
    double partialComputation,
    double normalizedStd,
    double sqrtDeltaT,
    std::atomic<double>& totalAverage,
    std::vector<std::vector<double>>& displayPaths,
    std::mutex& displayPathsMutex,
    bool collectDisplayPaths
) {
    std::vector<std::vector<double>> localDisplayPaths;
    double sumFinalPrices = 0.0;
    
    // Thread-local random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);
    
    for (int i = 0; i < numPaths; ++i) {
        double price = startingPrice;
        std::vector<double> path;
        
        if (collectDisplayPaths && localDisplayPaths.size() < 50) {
            path.reserve(steps);
            path.push_back(price);
        }
        
        // Simulate price path
        for (int j = 1; j < steps; ++j) {
            price *= std::exp(partialComputation + normalizedStd * sqrtDeltaT * d(gen));
            
            if (collectDisplayPaths && localDisplayPaths.size() < 50) {
                path.push_back(price);
            }
        }
        
        sumFinalPrices += price;
        
        if (collectDisplayPaths && localDisplayPaths.size() < 50) {
            localDisplayPaths.push_back(std::move(path));
        }
    }
    
    // Add local average to global atomic average
    double localAverage = sumFinalPrices / numPaths;
    AddToAtomic(totalAverage, localAverage);
    
    // Thread-safe transfer of display paths
    if (collectDisplayPaths && !localDisplayPaths.empty()) {
        std::lock_guard<std::mutex> lock(displayPathsMutex);
        displayPaths.insert(displayPaths.end(),
                          std::make_move_iterator(localDisplayPaths.begin()),
                          std::make_move_iterator(localDisplayPaths.end()));
    }
}

} // anonymous namespace

SimulationResult SimulateGBMMultiThreaded(
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
    
    // Determine thread count
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;
    
    // Distribute work across threads
    int pathsPerThread = paths / numThreads;
    int remainingPaths = paths % numThreads;
    
    std::atomic<double> totalAverage(0.0);
    std::vector<std::vector<double>> displayPaths;
    std::mutex displayPathsMutex;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    
    // Launch worker threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        int threadPaths = pathsPerThread + (i < static_cast<unsigned>(remainingPaths) ? 1 : 0);
        bool collectPaths = (i == 0);  // Only first thread collects display paths
        
        threads.emplace_back(
            SimulatePathsWorker,
            threadPaths,
            steps,
            startingPrice,
            partialComputation,
            normalizedStd,
            sqrtDeltaT,
            std::ref(totalAverage),
            std::ref(displayPaths),
            std::ref(displayPathsMutex),
            collectPaths
        );
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Calculate average across all threads
    double averagePredictedPrice = totalAverage / numThreads;
    
    return {displayPaths, averagePredictedPrice};
}

} // namespace gbm
