cmake_minimum_required(VERSION 3.10)
project(GBM)

# Set the pybind11 directory
set(pybind11_DIR "/home/andrew-keegan/Desktop/code/python/GBM/GBM/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Add subdirectory for C++ sources
add_subdirectory(src)