#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <random>
#include <vector>
#include <thread>
#include <atomic>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(int i, int j){
    return i+j;
}
void AddToAtomic(std::atomic<double>& atomicValue, double valueToAdd){
    double currentValue = atomicValue.load();
    while(!atomicValue.compare_exchange_weak(currentValue,currentValue+valueToAdd));
}
void simulatePaths(int numPaths, int steps, double startingPrice, double partialComputation, double normalizedStd, double sqrtDeltaT,
                   std::atomic<double>& totalAverage, std::vector<std::vector<double>>& displayPaths, bool collectDisplayPaths)
{
    std::vector<std::vector<double>> localDisplayPaths;
    double sumFinalPrices = 0.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0,1.0);
    
    for(int i=0; i<numPaths; ++i){
        std::vector<double> path(steps, startingPrice);
        double price = startingPrice;
        for(int j=1; j<steps;++j){
            price*=std::exp(partialComputation + normalizedStd * sqrtDeltaT * d(gen));
            path[j]=price;
        }
        sumFinalPrices+=price;
        if(collectDisplayPaths && i< 50){
            localDisplayPaths.push_back(std::move(path));
        }
    }
    double localAverage = sumFinalPrices / numPaths;
    AddToAtomic(totalAverage,localAverage);
    if(collectDisplayPaths){
        displayPaths = std::move(localDisplayPaths);
    }
}

void simulatePaths(int numPaths, int steps, double startingPrice, double partialComputation, double normalizedStd, double sqrtDeltaT,
                   std::atomic<double>& totalAverage, std::vector<std::vector<double>>& displayPaths, bool collectDisplayPaths);

std::pair<std::vector<std::vector<double>>, double> SimulateGBMMultiThreaded(double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,int steps, int totalPaths) {
    double deltaT = 1.0 / steps;
    double partialComputation = (normalizedMu - 0.5 * normalizedVar) * deltaT;
    double sqrtDeltaT = std::sqrt(deltaT);

    std::atomic<double> totalAverage(0.0);
    std::vector<std::vector<double>> displayPaths;
    std::mutex displayMutex;
    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int pathsPerThread = totalPaths / numThreads;
    int remainingPaths = totalPaths % numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int numPaths = pathsPerThread + (i < remainingPaths ? 1 : 0);
        threads.emplace_back(simulatePaths, numPaths, steps, startingPrice, partialComputation, normalizedStd, sqrtDeltaT,
                             std::ref(totalAverage), std::ref(displayPaths), (i == 0)); 
    }

    for (auto& thread : threads) {
        thread.join();
    }

    double averagePredictedPrice = totalAverage / numThreads;

    return {displayPaths, averagePredictedPrice};
}



std::pair<std::vector<std::vector<double>>,double> SimulatedGBM(double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,int steps, int paths){
    std::random_device rd;
    std::mt19937 gen(rd());
    double deltaT = 1.0/steps;
    std::normal_distribution<double> d(0.0,1.0);
    int displayPaths = 50;//could accept as param
    std::vector<std::vector<double>> fullPaths(std::min(paths, displayPaths));

    double sumFinalPrices = 0;
    double partialComputation = (normalizedMu - .5*normalizedVar) *deltaT;
    double sqrtDeltaT = std::sqrt(deltaT);

    for(int i = 0; i< paths; ++i){
        double price = startingPrice;
        std::vector<double> path;
        if(i<displayPaths){
            path.push_back(price);
        }
        for(int j=1;j<steps;++j){
            price*= std::exp(partialComputation + (normalizedStd * sqrtDeltaT*d(gen)));
            if(i<displayPaths){
                path.push_back(price);
            }
        }
        sumFinalPrices+=price;
        if(i<displayPaths){
            fullPaths[i]=path;
        }
    }
    double averagePredictedPrice = sumFinalPrices / paths;
    return {fullPaths, averagePredictedPrice};
} 
PYBIND11_MODULE(simulation, m) {
    m.doc() = "Simulation module for performing GBM simulations and calculating statistics"; // Module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("SimulatedGBM", &SimulatedGBM, "Simulate paths for Geometric Brownian Motion and calculate the average final price",
        py::arg("startingPrice"), py::arg("normalizedMu"), py::arg("normalizedVar"), py::arg("normalizedStd"), py::arg("steps"), py::arg("paths"));
    m.def("SimulateGBMMultiThreaded",&SimulateGBMMultiThreaded,"Simulate Paths for GBM using multiple threads",
        py::arg("startingPrice"), py::arg("normalizedMu"), py::arg("normalizedVar"), py::arg("normalizedStd"), py::arg("steps"), py::arg("paths"));
}