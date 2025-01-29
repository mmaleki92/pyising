cmake_minimum_required(VERSION 3.18)
project(Ising2DProject LANGUAGES CXX)

# ======================== pybind11 Setup ========================
# Option 1: Use submodule (requires proper initialization)
add_subdirectory(pybind11)

# Option 2: Fallback to find_package
# find_package(pybind11 REQUIRED)

# ======================== Core Configuration ========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ======================== Dependencies ========================
find_package(OpenMP REQUIRED)
add_subdirectory(cnpy)

# ======================== Ising Library ========================
add_library(ising STATIC src/ising.cpp src/ising.hpp)
target_link_libraries(ising PUBLIC cnpy OpenMP::OpenMP_CXX)
target_include_directories(ising PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

# ======================== Python Module ========================
pybind11_add_module(_pyising src/bindings.cpp)
target_link_libraries(_pyising PRIVATE ising pybind11::module OpenMP::OpenMP_CXX)

# ======================== Installation ========================
install(TARGETS _pyising LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ising ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})