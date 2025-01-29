cmake_minimum_required(VERSION 3.18)
project(Ising2DProject LANGUAGES CXX)

# ======================== Submodules ========================
add_subdirectory(pybind11)  # Local submodule
add_subdirectory(cnpy)      # Other dependency

# ======================== Core Setup ========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ======================== Dependencies ========================
find_package(OpenMP REQUIRED)

# ======================== Library Targets ========================
add_library(ising STATIC src/ising.cpp src/ising.hpp)
target_link_libraries(ising PUBLIC cnpy OpenMP::OpenMP_CXX)

# ======================== Python Module ========================
pybind11_add_module(_pyising src/bindings.cpp)
target_link_libraries(_pyising PRIVATE ising pybind11::module OpenMP::OpenMP_CXX)

# ======================== Installation ========================
install(TARGETS _pyising LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ising ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})