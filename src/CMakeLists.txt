pybind11_add_module(simulation cpp/simulation.cpp)
target_compile_options(simulation PRIVATE -mavx -O3 -march=native)