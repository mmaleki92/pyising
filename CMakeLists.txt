cmake_minimum_required(VERSION 3.10)
project(Ising2DProjectMPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# ====================================================================
# Safeguard to ensure the correct Conda compiler is being used.
# This prevents difficult-to-debug runtime library mismatch errors.
# ====================================================================
if(DEFINED ENV{CONDA_PREFIX})
    # We are inside a Conda environment. Let's check the compiler.
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
    endif()
endif()
# ====================================================================

# Prevent in-source builds (optional)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Please build in a separate directory.")
endif()

# Add subdirectories for external libraries
add_subdirectory(pybind11)
add_subdirectory(cnpy)

#---------------------
#  Add indicators here
#---------------------
option(INDICATORS_BUILD_EXAMPLES OFF) # Disable examples
option(INDICATORS_BUILD_TESTS OFF)    # Disable tests
add_subdirectory(indicators)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling OpenMP support.")
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP.")
endif()

# Find MPI
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI found. Enabling MPI support.")
else()
    message(FATAL_ERROR "MPI not found. Please install MPI.")
endif()

# Create a static library for the Ising2D implementation (MPI version)
add_library(ising STATIC
    src/ising.cpp
    src/ising.hpp
)
set_target_properties(ising PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ising PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/externals/pcg/include> # Add PCG include
)

# Optimization flags
target_compile_options(ising PRIVATE -O3 -march=native -fopenmp)
target_link_libraries(ising PUBLIC
    cnpy
    indicators
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    pybind11::headers
)

# Build the Python module using pybind11 (MPI version)
pybind11_add_module(_pyising src/bindings.cpp)
target_compile_options(_pyising PRIVATE -O3 -march=native -fopenmp)
target_link_libraries(_pyising PRIVATE ising OpenMP::OpenMP_CXX MPI::MPI_CXX)
set_target_properties(_pyising PROPERTIES INSTALL_RPATH "$ORIGIN")

# If using an older GNU compiler for <filesystem>:
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(ising PRIVATE stdc++fs)
    target_link_libraries(_pyising PRIVATE stdc++fs)
endif()


# Install rules
install(TARGETS cnpy LIBRARY DESTINATION .)
install(TARGETS _pyising LIBRARY DESTINATION .)
install(TARGETS ising ARCHIVE DESTINATION lib) 
