// Scalar (single-threaded) GBM simulation implementation
#include "simulation_common.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace gbm {

SimulationResult SimulateGBMScalar(
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
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);
    
    // Prepare display paths (first 50 paths)
    const int displayPathsCount = std::min(50, paths);
    std::vector<std::vector<double>> displayPaths;
    displayPaths.reserve(displayPathsCount);
    
    double sumFinalPrices = 0.0;
    
    // Simulate paths
    for (int i = 0; i < paths; ++i) {
        double price = startingPrice;
        std::vector<double> path;
        
        // Collect path data for display paths
        if (i < displayPathsCount) {
            path.reserve(steps);
            path.push_back(price);
        }
        
        // Simulate price path
        for (int j = 1; j < steps; ++j) {
            double noise = d(gen);
            price *= std::exp(partialComputation + normalizedStd * sqrtDeltaT * noise);
            
            if (i < displayPathsCount) {
                path.push_back(price);
            }
        }
        
        sumFinalPrices += price;
        
        if (i < displayPathsCount) {
            displayPaths.push_back(std::move(path));
        }
    }
    
    double averagePredictedPrice = sumFinalPrices / paths;
    
    return {displayPaths, averagePredictedPrice};
}

} // namespace gbm
