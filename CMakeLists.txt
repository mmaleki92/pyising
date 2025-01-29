cmake_minimum_required(VERSION 3.10)
project(Ising2DProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add this near the top after setting C++ standard
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # <-- Crucial for static libs used in shared modules

# Prevent in-source builds (optional best practice)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Please build in a separate directory.")
endif()

# Find pybind11 (either installed or via submodule)
find_package(pybind11 REQUIRED)

# Add cnpy library (assuming it's in a subdirectory "cnpy")
add_subdirectory(cnpy)
option(BUILD_EXECUTABLE "Build the ising_demo executable" ON)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling OpenMP support.")
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP.")
endif()

# Create a static library for the Ising2D implementation
add_library(ising STATIC
    src/ising.cpp
    src/ising.hpp
)
# Add explicit PIC flag for static library (redundant but safe)
set_target_properties(ising PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Include directories
target_include_directories(ising PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link libraries: cnpy and OpenMP
target_link_libraries(ising PRIVATE cnpy OpenMP::OpenMP_CXX)

# Build the Python module using pybind11
pybind11_add_module(pyising src/bindings.cpp)

# Link OpenMP to the pybind11 module
target_link_libraries(pyising PRIVATE ising OpenMP::OpenMP_CXX)

# Optional: build an executable for a demo or CLI
# add_executable(ising_demo main.cpp)
# target_link_libraries(ising_demo PRIVATE ising OpenMP::OpenMP_CXX)

# If using an older GNU Compiler (pre-9.x), link stdc++fs for <filesystem> support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(ising PRIVATE stdc++fs)
    target_link_libraries(pyising PRIVATE stdc++fs)
endif()


# Install targets
install(TARGETS ising ARCHIVE DESTINATION lib)
install(TARGETS pyising LIBRARY DESTINATION pyising)

# # Conditionally build executable
# if(BUILD_EXECUTABLE)
#     add_executable(ising_demo main.cpp)
#     target_link_libraries(ising_demo PRIVATE ising OpenMP::OpenMP_CXX)
#     install(TARGETS ising_demo RUNTIME DESTINATION bin)
# endif()