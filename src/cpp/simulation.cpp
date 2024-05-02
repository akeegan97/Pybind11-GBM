#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <random>
#include <vector>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <functional>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(int i, int j){
    return i+j;
}
void AddToAtomic(std::atomic<double>& atomicValue, double valueToAdd){
    double currentValue = atomicValue.load();
    while(!atomicValue.compare_exchange_weak(currentValue,currentValue+valueToAdd));
}
//SIMD custom function for computing exponential function fitted to range (-0.2,0.2)
//error compared to STD::EXP() over same range ~ (7.5E-09:3.1E-10)
__m256d exp_approx(__m256d x) {
    //coefficients found using numpy and scipy.optimize for the range 
    __m256d c0 = _mm256_set1_pd(1);
    __m256d c1 = _mm256_set1_pd(1);
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
double CalculateSIMDPaths(int numPaths, int steps, double startingPrice, double partialComputation, double normalizedStd, double sqrtDeltaT)
{
    //caluations per step in path
    /*
    steps
    loop: 
        x = price
        a = normStd * sqrtDT
        b = a * randomval
        c = partialcomp + b
        d = exp_approx(c)
        x = x*d
    finalPrice = x 
    */
    double averageForThisThread = 0;
    double finalPrices[4];
    __m256d _zeroes = _mm256_setzero_pd();
    __m256d _normalDistrValues;

    //constants
    __m256d _normalStdVec = _mm256_set1_pd(normalizedStd);
    __m256d _partialCompVec = _mm256_set1_pd(partialComputation);
    __m256d _sqrtDTVec = _mm256_set1_pd(sqrtDeltaT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0,1.0);

    for(int i=0; i<numPaths;i+=4){
        alignas(32) double prices[4];  
        for(int k=0; k<4;++k){
            prices[k]=startingPrice;
        }
        __m256d _prices = _mm256_load_pd(prices);
        
        for(int j =1;j<steps;++j){
            alignas(32) double ranNums[4];
            for(int k=0;k<4;++k){
                ranNums[k]= d(gen);
            }
            _normalDistrValues = _mm256_loadu_pd(ranNums);
            //compute 
            __m256d _a = _mm256_mul_pd(_normalStdVec,_sqrtDTVec);
            __m256d _c = _mm256_fmadd_pd(_a,_normalDistrValues,_partialCompVec);
            __m256d _d = exp_approx(_c);
            _prices = _mm256_mul_pd(_prices,_d);
        }
        _mm256_storeu_pd(finalPrices,_prices);
        double averageForThisPass = 0;
        for(int k=0;k<4;k++){
            averageForThisPass+=finalPrices[k];
        }
        averageForThisThread+= (averageForThisPass);
    }
    return averageForThisThread / numPaths;
}

std::pair<std::vector<std::vector<double>>, double> SimulateGBMMultiThreaded(double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,int steps, int totalPaths) {
    double deltaT = 1.0 / steps;
    double partialComputation = (normalizedMu - 0.5 * normalizedVar) * deltaT;
    double sqrtDeltaT = std::sqrt(deltaT);

    std::atomic<double> totalAverage(0.0);
    std::vector<std::vector<double>> displayPaths;

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

std::pair<std::vector<std::vector<double>>,double> SimulateGBMIntrinsicMT(double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,int steps, int totalPaths){
    double deltaT = 1.0 / steps;
    double partialComputation = (normalizedMu - 0.5 * normalizedVar) * deltaT;
    double sqrtDeltaT = std::sqrt(deltaT);

    std::vector<std::vector<double>> displayPaths;
    int displayPathsCount = std::min(50, totalPaths);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0,1.0);
    for (int i = 0; i < displayPathsCount; ++i) {
        std::vector<double> path(steps, startingPrice);
        double price = startingPrice;
        for (int j = 1; j < steps; ++j) {
            double noise = d(gen);
            price *= std::exp(partialComputation + normalizedStd * sqrtDeltaT * noise);
            path[j] = price;
        }
        displayPaths.push_back(std::move(path));
    }

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int pathsPerThread = totalPaths / numThreads;
    int remainingPaths = totalPaths % numThreads;

    std::vector<double> averagePrices(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        int numPaths = pathsPerThread + (i < remainingPaths ? 1 : 0);
        threads.emplace_back([&averagePrices, i, numPaths, steps, startingPrice, partialComputation, normalizedStd, sqrtDeltaT]() {
            double averagePrice = CalculateSIMDPaths(numPaths, steps, startingPrice, partialComputation, normalizedStd, sqrtDeltaT);
            averagePrices[i] = averagePrice;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    double totalAveragePrice = 0.0;
    for (double price : averagePrices) {
        totalAveragePrice += price;
    }
    totalAveragePrice /= numThreads;

    return {displayPaths, totalAveragePrice};

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
    m.def("SimulateGBMIntrinsicMT",&SimulateGBMIntrinsicMT,"Using SIMD instructions",
        py::arg("startingPrice"), py::arg("normalizedMu"), py::arg("normalizedVar"), py::arg("normalizedStd"), py::arg("steps"), py::arg("paths"));
}