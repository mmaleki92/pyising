cmake_minimum_required(VERSION 3.10)
project(Ising2DProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 (either installed or via submodule)
find_package(pybind11 REQUIRED)

# Library containing our Ising2D implementation
add_library(ising STATIC
    ising.cpp
    ising.hpp
)

target_include_directories(ising PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Python module
pybind11_add_module(pyising bindings.cpp)
target_link_libraries(pyising PRIVATE ising)

# Optional: build an executable for a demo
add_executable(ising_demo main.cpp)
target_link_libraries(ising_demo PRIVATE ising)