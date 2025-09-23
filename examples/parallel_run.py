from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pyising
import numpy as np
import os
import time

# Create output directory if it doesn't exist
output_dir = "simulation"
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
comm.Barrier()  # Make sure directory exists before all ranks proceed

# Configure your simulation parameters
temps = np.linspace(1, 4, 50)
L = 64
N_steps = 10000
equ_N = 1000
snapshot_interval = 100
seed_base = 42
use_wolff = False
save_all_configs = True

print(f"Rank {rank}: Starting simulation with L={L}, {len(temps)} temperatures")
start_time = time.time()

# Run the parallel simulation
results = pyising.run_parallel_metropolis(
    temps, L, N_steps=N_steps,
    equ_N=equ_N,
    snapshot_interval=snapshot_interval,
    seed_base=seed_base,
    output_dir=output_dir,
    use_wolff=use_wolff, 
    save_all_configs=save_all_configs
)

end_time = time.time()
print(f"Rank {rank}: Finished simulation in {end_time - start_time:.2f} seconds")

# If you need to aggregate results from all ranks at the end
all_results = comm.gather(results, root=0)

if rank == 0:
    # Process or save combined results (but not loading all configurations)
    flat_results = []
    for res_list in all_results:
        flat_results.extend(res_list)
    
    # Now you can analyze the combined results
    print(f"Total results collected: {len(flat_results)}")
    
    # Save the aggregated results (without configurations)
    temps_array = np.array([res.T for res in flat_results])
    mag_array = np.array([res.mean_mag for res in flat_results])
    energy_array = np.array([res.mean_ene for res in flat_results])
    binder_array = np.array([res.binder for res in flat_results])
    
    # Sort by temperature for plotting
    sort_idx = np.argsort(temps_array)
    temps_array = temps_array[sort_idx]
    mag_array = mag_array[sort_idx]
    energy_array = energy_array[sort_idx]
    binder_array = binder_array[sort_idx]
    
    # Save aggregated data
    results_file = os.path.join(output_dir, f"L_{L}_results.txt")
    with open(results_file, 'w') as f:
        f.write("# Temperature Magnetization Energy Binder\n")
        for i in range(len(temps_array)):
            f.write(f"{temps_array[i]} {mag_array[i]} {energy_array[i]} {binder_array[i]}\n")
    
    print(f"Saved aggregated results to {results_file}")