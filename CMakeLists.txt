cmake_minimum_required(VERSION 3.18)
project(Ising2DProject LANGUAGES CXX)

# ======================== Pybind11 Setup (Submodule) ========================
add_subdirectory(pybind11)  # Uses local submodule
# Remove all find_package(pybind11) calls
# No need for include_directories() - pybind11 handles this

# ======================== Core Configuration ========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ======================== Dependencies ========================
find_package(OpenMP REQUIRED)
add_subdirectory(cnpy)  # Assuming cnpy is also a submodule

# ======================== Ising Library ========================
add_library(ising STATIC
    src/ising.cpp
    src/ising.hpp
)
target_link_libraries(ising PUBLIC 
    cnpy
    OpenMP::OpenMP_CXX
)

# ======================== Python Module ========================
pybind11_add_module(_pyising src/bindings.cpp)  # Use pybind11's helper
target_link_libraries(_pyising PRIVATE
    ising
    pybind11::module  # Link to pybind11's CMake target
    OpenMP::OpenMP_CXX
)

# ======================== Installation ========================
install(TARGETS _pyising
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # Use standard CMake variable
)

install(TARGETS ising
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)