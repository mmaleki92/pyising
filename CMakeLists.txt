cmake_minimum_required(VERSION 3.10)
project(Ising2DProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prevent in-source builds (optional)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Please build in a separate directory.")
endif()

# Option to enable MPI
option(USE_MPI "Build MPI-enabled version" OFF)

# Add subdirectories for external libraries
add_subdirectory(pybind11)
add_subdirectory(cnpy)
add_subdirectory(indicators)

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
set_target_properties(ising PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ising PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/externals/pcg/include> # If you're using PCG
)
target_compile_options(ising PRIVATE -O3 -march=native -fopenmp)

target_link_libraries(ising PUBLIC
    cnpy
    indicators
    OpenMP::OpenMP_CXX
)

# Build the Python module - serial version (always built)
pybind11_add_module(_pyising
    src/bindings.cpp
)
target_compile_options(_pyising PRIVATE -O3 -march=native -fopenmp)
target_link_libraries(_pyising PRIVATE ising OpenMP::OpenMP_CXX)
set_target_properties(_pyising PROPERTIES INSTALL_RPATH "$ORIGIN")

# If using an older GNU compiler for <filesystem>:
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(ising PRIVATE stdc++fs)
    target_link_libraries(_pyising PRIVATE stdc++fs)
endif()

# If the user requests MPI, find MPI and build an MPI-enabled version of the module
if(USE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message(STATUS "MPI found. Building MPI-enabled PyBind11 module.")
       
        # Create a new target with MPI flags
        add_library(ising_mpi STATIC
            src/ising.cpp
            src/ising.hpp
        )
        set_target_properties(ising_mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_options(ising_mpi PRIVATE -O3 -march=native -fopenmp)
        target_include_directories(ising_mpi PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(ising_mpi PUBLIC
            cnpy
            indicators
            OpenMP::OpenMP_CXX
            MPI::MPI_CXX
        )

        pybind11_add_module(_pyising_mpi
            src/bindings.cpp
        )
        target_compile_options(_pyising_mpi PRIVATE -O3 -march=native -fopenmp)
        target_link_libraries(_pyising_mpi PRIVATE
            ising_mpi
            OpenMP::OpenMP_CXX
            MPI::MPI_CXX
        )
        set_target_properties(_pyising_mpi PROPERTIES INSTALL_RPATH "$ORIGIN")

        # If using older <filesystem> needs:
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
            target_link_libraries(ising_mpi PRIVATE stdc++fs)
            target_link_libraries(_pyising_mpi PRIVATE stdc++fs)
        endif()

        # Install MPI version as well
        install(TARGETS _pyising_mpi LIBRARY DESTINATION .)
        install(TARGETS ising_mpi ARCHIVE DESTINATION lib)
    else()
        message(WARNING "MPI requested but not found. Only the serial version will be built.")
    endif()
endif()

# Install rules for the serial version
install(TARGETS cnpy LIBRARY DESTINATION .)
install(TARGETS _pyising LIBRARY DESTINATION .)
install(TARGETS ising ARCHIVE DESTINATION lib)