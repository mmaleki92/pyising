from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pyising
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = "simulation"
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
comm.Barrier()  # Make sure directory exists before all ranks proceed

temps = np.linspace(1, 4, 50)

L = 64
results = pyising.run_parallel_metropolis(temps, L, N_steps=10000,
                                               equ_N=1000,
                                               snapshot_interval=100,
                                               seed_base=42,
                                               output_dir=output_dir,
                                               use_wolff=False, 
                                               save_all_configs=True)

# If you need to aggregate results from all ranks at the end
all_results = comm.gather(results, root=0)

if rank == 0:
    # Process or save combined results
    flat_results = []
    for res_list in all_results:
        flat_results.extend(res_list)
    
    # Now you can analyze the combined results
    print(f"Total results collected: {len(flat_results)}")