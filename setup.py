from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "ising",
        ["ising.cpp"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name="ising",
    ext_modules=ext_modules,
)