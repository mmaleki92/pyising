from skbuild import setup

setup(
    name="pyising",
    version="0.1.0",
    description="2D Ising model simulation with Python bindings",
    author="Morteza Maleki",
    license="MIT",
    packages=["pyising"],
    package_dir={"pyising": "pyising"},
    python_requires=">=3.6",
    install_requires=["numpy"],
    setup_requires=["scikit-build", "cmake>=3.10", "ninja", "pybind11>=2.6"],
    cmake_args=["-DBUILD_EXECUTABLE=OFF"],
    include_package_data=True,
)