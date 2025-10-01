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
temps = np.linspace(1.5, 3.5, 60) # Focused temperature range for better resolution around Tc
L = 32
N_steps = 20000  # Number of measurement steps
equ_N = 5000     # Number of equilibration steps
snapshot_interval = 2000
seed_base = 42
use_wolff = True # Wolff is more efficient near the critical point
save_all_configs = False # Set to False if you just need the plots

if rank == 0:
    print(f"Starting simulation with L={L}, {len(temps)} temperatures using {size} MPI ranks.")
    print(f"Algorithm: {'Wolff' if use_wolff else 'Metropolis'}")
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

# --- Data Aggregation and Processing (on Rank 0) ---
# Gather results from all ranks to the root process (rank 0)
all_results = comm.gather(local_results, root=0)

if rank == 0:
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    
    # Process or save combined results
    flat_results = []
    if all_results:
        for res_list in all_results:
            if res_list:  # Ensure the list from a rank is not empty
                flat_results.extend(res_list)

    print(f"Total results collected: {len(flat_results)}")

    # --- UPDATED: Extract all data into NumPy arrays ---
    temps_array = np.array([res['T'] for res in flat_results])
    mag_array = np.array([res['mean_mag'] for res in flat_results])
    energy_array = np.array([res['mean_ene'] for res in flat_results])
    binder_array = np.array([res['binder'] for res in flat_results])
    susceptibility_array = np.array([res['susceptibility'] for res in flat_results])
    specific_heat_array = np.array([res['specific_heat'] for res in flat_results])
    correlation_length_array = np.array([res['correlation_length'] for res in flat_results])

    # Sort by temperature for clean plotting
    sort_idx = np.argsort(temps_array)
    temps_array = temps_array[sort_idx]
    mag_array = mag_array[sort_idx]
    energy_array = energy_array[sort_idx]
    binder_array = binder_array[sort_idx]
    susceptibility_array = susceptibility_array[sort_idx]
    specific_heat_array = specific_heat_array[sort_idx]
    correlation_length_array = correlation_length_array[sort_idx]

    # --- UPDATED: Save aggregated data to a text file ---
    results_file = os.path.join(output_dir, f"L_{L}_results.txt")
    header = "# Temperature Magnetization Energy Binder Susceptibility SpecificHeat CorrLength"
    data_to_save = np.vstack([
        temps_array, mag_array, energy_array, binder_array, 
        susceptibility_array, specific_heat_array, correlation_length_array
    ]).T
    np.savetxt(results_file, data_to_save, header=header)
    print(f"Saved aggregated results to {results_file}")

    # --- REVISED: Plotting Section ---
    print("Generating plots...")
    
    # Create a figure with a 3x2 grid of subplots for better layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Ising Model Physical Properties (L={L})', fontsize=16)

    # Plot 1: Magnetization
    axes[0, 0].plot(temps_array, mag_array, 'o-', markersize=4, label='$\\langle|m|\\rangle$')
    axes[0, 0].set_ylabel('Magnetization $\\langle|m|\\rangle$')
    axes[0, 0].grid(True)

    # Plot 2: Energy
    axes[0, 1].plot(temps_array, energy_array, 's-', color='red', markersize=4, label='$\\langle E \\rangle$')
    axes[0, 1].set_ylabel('Energy $\\langle E \\rangle$')
    axes[0, 1].grid(True)

    # Plot 3: Susceptibility
    axes[1, 0].plot(temps_array, susceptibility_array, 'D-', color='purple', markersize=4, label='$\\chi$')
    axes[1, 0].set_ylabel('Susceptibility $\\chi$')
    axes[1, 0].grid(True)

    # Plot 4: Specific Heat
    axes[1, 1].plot(temps_array, specific_heat_array, 'v-', color='orange', markersize=4, label='$C_V$')
    axes[1, 1].set_ylabel('Specific Heat $C_V$')
    axes[1, 1].grid(True)
    
    # Plot 5: Binder Cumulant
    axes[2, 0].plot(temps_array, binder_array, '^-', color='green', markersize=4, label='$U_L$')
    axes[2, 0].set_ylabel('Binder Cumulant $U_L$')
    axes[2, 0].set_xlabel('Temperature $T$')
    axes[2, 0].grid(True)

    # Plot 6: Correlation Length
    axes[2, 1].plot(temps_array, correlation_length_array, '*-', color='brown', markersize=4, label='$\\xi$')
    axes[2, 1].set_ylabel('Correlation Length $\\xi$')
    axes[2, 1].set_xlabel('Temperature $T$')
    axes[2, 1].grid(True)
    
    # Add a vertical line for the theoretical critical temperature
    Tc_exact = 2 / np.log(1 + np.sqrt(2))
    for ax_row in axes:
        for ax in ax_row:
            ax.axvline(x=Tc_exact, color='gray', linestyle='--', linewidth=2, label=f'$T_c$ (exact)')
    
    # Improve legends
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles_tc, labels_tc = axes[0,0].get_legend_handles_labels()
    if '$\\langle|m|\\rangle$' in labels:
        idx = labels.index('$\\langle|m|\\rangle$')
        del handles[idx]
        del labels[idx]

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_file = os.path.join(output_dir, f"L_{L}_plots.png")
    plt.savefig(plot_file)
    print(f"Saved plots to {plot_file}")
    
    # plt.show()