cmake_minimum_required(VERSION 3.10)
project(Ising2DProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 (either installed or via submodule)
find_package(pybind11 REQUIRED)

# Add cnpy (assuming it's in a subdirectory named "cnpy")
add_subdirectory(cnpy)

# Library containing our Ising2D implementation
add_library(ising STATIC
    ising.cpp
    ising.hpp
)

target_include_directories(ising PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link cnpy to the ising library
target_link_libraries(ising PRIVATE cnpy)

# Python module
pybind11_add_module(pyising bindings.cpp)
target_link_libraries(pyising PRIVATE ising cnpy)

# Optional: build an executable for a demo
add_executable(ising_demo main.cpp)
target_link_libraries(ising_demo PRIVATE ising)

# Link stdc++fs for <filesystem> support (if needed)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(ising PRIVATE stdc++fs)
    target_link_libraries(pyising PRIVATE stdc++fs)
    target_link_libraries(ising_demo PRIVATE stdc++fs)
endif()
