from skbuild import setup  # Use scikit-build
from setuptools import find_packages

setup(
    name="pyising",
    version="0.1.3",
    author="Morteza Maleki",
    author_email="maleki.morteza92@gmail.com",
    description="Python bindings for the Ising2D C++ simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mmaleki2/pyising",
    packages=find_packages(where="."),
    cmake_install_dir="pyising",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pybind11>=2.5.0",
        "mpi4py",
        "numpy",
        "matplotlib",
        "scipy"
    ],
)