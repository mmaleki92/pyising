import pyising
import numpy as np

temps = np.linspace(1, 4, 100)

L = 64
results = pyising.run_parallel_metropolis(temps, L, N_steps=10000,
                                                   seed_base=42,
                                                   output_dir="simultion",
                                                    use_wolff=False, 
                                                    save_all_configs=True)
