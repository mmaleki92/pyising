Install pybind11 (for example, using your system package manager or pip install pybind11).
Create a folder (e.g., "build") inside your project directory and navigate there.
Run:
cmake ..
then
make
This will produce the shared library (pyising.so on Linux/macOS or pyising.pyd on Windows).
Copy or link the resulting module into your Python environment or run Python from the build directory.
Run the Python script:
python ../example.py