from mpi4py import MPI
import pyising
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Simulation Setup ---
# Create output directory if it doesn't exist
output_dir = "simulation"
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
comm.Barrier()  # Make sure directory exists before all ranks proceed

# Configure your simulation parameters
temps = np.linspace(1, 4, 50)
L = 128
N_steps = 1000000
equ_N = 10000
snapshot_interval = 1000
seed_base = 42
use_wolff = False
save_all_configs = True

if rank == 0:
    print(f"Rank {rank}: Starting simulation with L={L}, {len(temps)} temperatures")
    start_time = time.time()

# Run the parallel simulation
# This returns a list of dictionaries on each rank
local_results = pyising.run_parallel_metropolis(
    temps, L, N_steps=N_steps,
    equ_N=equ_N,
    snapshot_interval=snapshot_interval,
    seed_base=seed_base,
    output_dir=output_dir,
    use_wolff=use_wolff,
    save_all_configs=save_all_configs
)

if rank == 0:
    end_time = time.time()
    print(f"Rank {rank}: Finished simulation in {end_time - start_time:.2f} seconds")

# --- Data Aggregation and Processing (on Rank 0) ---
# Gather results from all ranks to the root process (rank 0)
all_results = comm.gather(local_results, root=0)

if rank == 0:
    # Process or save combined results
    flat_results = []
    for res_list in all_results:
        if res_list:  # Ensure the list from a rank is not empty
            flat_results.extend(res_list)

    print(f"Total results collected: {len(flat_results)}")

    # Extract data into NumPy arrays
    temps_array = np.array([res['T'] for res in flat_results])
    mag_array = np.array([res['mean_mag'] for res in flat_results])
    energy_array = np.array([res['mean_ene'] for res in flat_results])
    binder_array = np.array([res['binder'] for res in flat_results])

    # Sort by temperature for clean plotting
    sort_idx = np.argsort(temps_array)
    temps_array = temps_array[sort_idx]
    mag_array = mag_array[sort_idx]
    energy_array = energy_array[sort_idx]
    binder_array = binder_array[sort_idx]

    # Save aggregated data to a text file
    results_file = os.path.join(output_dir, f"L_{L}_results.txt")
    with open(results_file, 'w') as f:
        f.write("# Temperature Magnetization Energy Binder\n")
        for i in range(len(temps_array)):
            f.write(f"{temps_array[i]} {mag_array[i]} {energy_array[i]} {binder_array[i]}\n")
    print(f"Saved aggregated results to {results_file}")

    # --- NEW: Plotting Section ---
    print("Generating plots...")
    
    # Create a figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    
    # Plot 1: Absolute Magnetization vs. Temperature
    axes[0].plot(temps_array, mag_array, 'o-', label=f'L={L}')
    axes[0].set_ylabel('Absolute Magnetization $|M|$')
    axes[0].set_title(f'Ising Model Physical Properties (L={L})')
    axes[0].grid(True)
    
    # Plot 2: Energy vs. Temperature
    axes[1].plot(temps_array, energy_array, 's-', color='red', label=f'L={L}')
    axes[1].set_ylabel('Energy per site $\\langle E \\rangle$')
    axes[1].grid(True)

    # Plot 3: Binder Cumulant vs. Temperature
    axes[2].plot(temps_array, binder_array, '^-', color='green', label=f'L={L}')
    axes[2].set_ylabel('Binder Cumulant $U_L$')
    axes[2].set_xlabel('Temperature $T$')
    axes[2].grid(True)

    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout()
    
    # Save the figure to a file
    plot_file = os.path.join(output_dir, f"L_{L}_plots.png")
    plt.savefig(plot_file)
    print(f"Saved plots to {plot_file}")
    
    # Optionally, display the plot interactively
    # plt.show()