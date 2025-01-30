cmake_minimum_required(VERSION 3.10)
project(Ising2DProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prevent in-source builds (optional)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Please build in a separate directory.")
endif()

# Add subdirectories for external libraries
add_subdirectory(pybind11)
add_subdirectory(cnpy)

# ────────────────────
#  Add indicators here
# ────────────────────
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

# Create a static library for the Ising2D implementation
add_library(ising STATIC
    src/ising.cpp
    src/ising.hpp
)
set_target_properties(ising PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ising PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link libraries: cnpy, OpenMP, and now indicators
target_link_libraries(ising PUBLIC
    cnpy
    indicators        # Link the indicators library
    OpenMP::OpenMP_CXX
)

# Build the Python module using pybind11
pybind11_add_module(_pyising src/bindings.cpp)
target_link_libraries(_pyising PRIVATE ising OpenMP::OpenMP_CXX)
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