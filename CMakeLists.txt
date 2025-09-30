cmake_minimum_required(VERSION 3.10)
project(Ising2DProjectMPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# [cite_start]Safeguard to ensure the correct Conda compiler is being used. [cite: 2]
# [cite_start]This prevents difficult-to-debug runtime library mismatch errors. [cite: 2]
# ====================================================================
if(DEFINED ENV{CONDA_PREFIX})
    # We are inside a Conda environment. [cite_start]Let's check the compiler. [cite: 3]
    string(FIND "${CMAKE_CXX_COMPILER}" "$ENV{CONDA_PREFIX}" IS_CONDA_COMPILER)

    if(IS_CONDA_COMPILER EQUAL -1)
        # The found compiler is NOT inside the Conda environment path.
        message(FATAL_ERROR "
        #####################################################################
        # ERROR: Incorrect C++ compiler detected!                           #
        #####################################################################
        You are in a Conda environment, but CMake is using the system's
        C++ compiler:
          ${CMAKE_CXX_COMPILER}

        This will lead to a GLIBCXX ImportError at runtime.

        To fix this, please re-run the installation command specifying
        the correct Conda compilers, for example:

        CXX=x86_64-conda-linux-gnu-g++ CC=x86_64-conda-linux-gnu-gcc pip install .
        #####################################################################
        ")
    [cite_start]endif() [cite: 4]
endif()
# ====================================================================

# Prevent in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Please build in a separate directory.")
endif()

# Add subdirectories for external libraries
add_subdirectory(pybind11)
add_subdirectory(cnpy)

#---------------------
#  Add indicators library
#---------------------
option(INDICATORS_BUILD_EXAMPLES OFF)
option(INDICATORS_BUILD_TESTS OFF)
add_subdirectory(indicators)

# Find necessary packages
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    [cite_start]message(STATUS "OpenMP found. Enabling OpenMP support.") [cite: 5]
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP.")
endif()

find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI found. Enabling MPI support.")
else()
    message(FATAL_ERROR "MPI not found. Please install MPI.")
endif()

# ====================================================================
# Build the Python module, compiling all sources directly into it.
# This avoids creating an unnecessary intermediate static library and
# simplifies include path management.
# ====================================================================
pybind11_add_module(_pyising
    src/bindings.cpp
    src/ising.cpp
)

# Add include directories for the module's dependencies
target_include_directories(_pyising PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/pcg/include
    ${CMAKE_SOURCE_DIR}/cnpy    
    ${CMAKE_SOURCE_DIR}/indicators/include 
)

# Add compile options for the module
target_compile_options(_pyising PRIVATE -O3 -march=native -fopenmp)

# Link the module against all of its dependencies
[cite_start]target_link_libraries(_pyising PRIVATE [cite: 6]
    cnpy
    indicators
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

set_target_properties(_pyising PROPERTIES INSTALL_RPATH "$ORIGIN")

# If using an older GNU compiler for <filesystem>
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(_pyising PRIVATE stdc++fs)
endif()

# Install the final Python module and its dependencies
install(TARGETS cnpy LIBRARY DESTINATION .)
install(TARGETS _pyising LIBRARY DESTINATION .)