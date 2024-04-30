#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <random>
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(int i, int j){
    return i+j;
}
std::pair<std::vector<std::vector<double>>,double> SimulatedGBM(double startingPrice, double normalizedMu, double normalizedVar, double normalizedStd,int steps, int paths){
    std::random_device rd;
    std::mt19937 gen(rd());
    double deltaT = 1.0/steps;
    std::normal_distribution<> d(0,1);
    int displayPaths = 50;//could accept as param
    std::vector<std::vector<double>> fullPaths(std::min(paths, displayPaths));

    double sumFinalPrices = 0;
    double partialComputation = (normalizedMu - .5*normalizedVar) *deltaT;

    for(int i = 0; i< paths; ++i){
        double price = startingPrice;
        std::vector<double> path;
        if(i<displayPaths){
            path.push_back(price);
        }
        for(int j=1;j<steps;++j){
            price*= std::exp(partialComputation + (normalizedStd * std::sqrt(deltaT)*d(gen)));
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

}