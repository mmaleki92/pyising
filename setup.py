from skbuild import setup
from setuptools import find_packages

setup(
    name="pyising",
    version="0.1.0",
    author="Morteza Maleki",
    author_email="maleki.morteza92@gmail.com",
    description="Python bindings for the Ising2D C++ simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Ising2DProject",
    packages=find_packages(where="."),
    cmake_install_dir="pyising",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy>=1.22.0"],  # pybind11 is already in-tree
    cmake_args=[
        "-DCMAKE_VERBOSE_MAKEFILE=ON",  # Optional for debugging
    ],
)