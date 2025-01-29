from build import pyising
import numpy as np

temps = np.linspace(1, 4, 100)
results = pyising.run_parallel_metropolis(temps, L=32, N_steps=1000, seed_base=42)

for temp, result in zip(temps, results):
    print(f"Temp: {temp}, Binder: {result.binder}, Magnetization: {result.meanMag}")