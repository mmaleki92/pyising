[![Upload Python Package](https://github.com/mmaleki92/pyising/actions/workflows/python-publish.yml/badge.svg)](https://github.com/mmaleki92/pyising/actions/workflows/python-publish.yml)

# PyIsing

PyIsing is a Python library for simulating the Ising model using C++ bindings via pybind11. It includes implementations of the Wolff and Metropolis Monte Carlo algorithms for efficient simulation of spin systems.

## Features
- Fast C++ core: Uses pybind11 to leverage high-performance C++ computations.
- Wolff Algorithm: Implements cluster updates for efficient decorrelation.
- Metropolis Algorithm: Standard single-spin update method.
- Flexible Parameters: Supports different lattice sizes, temperatures.

## installation
```bash
pip install pyising
```

for buliding from scratch

```bash
pythin setup.py install
```
## usage
```python
import pyising

temps = np.linspace(1, 4, 50)

L = 64
results = pyising.run_parallel_metropolis(temps, L, N_steps=100000,
                                                equ_N=10000,
                                                seed_base=42,
                                                output_dir="simultion",
                                                use_wolff=False, 
                                                save_all_configs=False)
```


# References
I built the code based on the following code, added the python binding and made it modular after refining the structure.
https://github.com/VictorSeven/IsingModel
