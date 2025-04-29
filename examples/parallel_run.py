from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pyising
import numpy as np

temps = np.linspace(1, 4, 50)

L = 64
results = pyising.run_parallel_metropolis(temps, L, N_steps=10000,
                                                equ_N=1000,
                                                snapshot_interval=100,
                                                seed_base=42,
                                                output_dir="simultion",
                                                use_wolff=False, 
                                                save_all_configs=True)
