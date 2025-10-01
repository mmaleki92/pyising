from mpi4py import MPI
import pyising
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- MPI and Simulation Setup (remains the same) ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

output_dir = "simulation_results"
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
comm.Barrier()

temps = np.linspace(1.5, 3.5, 60)
L = 32
N_steps = 20000  # Number of measurement steps
equ_N = 5000     # Number of equilibration steps
snapshot_interval = 2000
seed_base = 42
use_wolff = True # Wolff is more efficient near the critical point
save_all_configs = False # Set to False if you just need the plots

if rank == 0:
    print(f"Starting simulation with L={L}, {len(temps)} temperatures...")
    start_time = time.time()

local_results = pyising.run_parallel_metropolis(
    temps, L, N_steps=N_steps,
    equ_N=equ_N,
    snapshot_interval=snapshot_interval,
    seed_base=seed_base,
    output_dir=output_dir,
    use_wolff=use_wolff, 
    save_all_configs=save_all_configs
)
all_results = comm.gather(local_results, root=0)

if rank == 0:
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    
    flat_results = []
    if all_results:
        for res_list in all_results:
            if res_list: flat_results.extend(res_list)

    print(f"Total results collected: {len(flat_results)}")

    # Extract primary data
    temps_array = np.array([res['T'] for res in flat_results])
    mag_array = np.array([res['mean_mag'] for res in flat_results])
    energy_array = np.array([res['mean_ene'] for res in flat_results])
    binder_array = np.array([res['binder'] for res in flat_results])
    susceptibility_array = np.array([res['susceptibility'] for res in flat_results])
    specific_heat_array = np.array([res['specific_heat'] for res in flat_results])
    
    # Sort everything by temperature
    sort_idx = np.argsort(temps_array)
    temps_array = temps_array[sort_idx]
    mag_array = mag_array[sort_idx]
    energy_array = energy_array[sort_idx]
    binder_array = binder_array[sort_idx]
    susceptibility_array = susceptibility_array[sort_idx]
    specific_heat_array = specific_heat_array[sort_idx]
    sorted_results = [flat_results[i] for i in sort_idx]

    # --- NEW: Calculate Correlation Length by fitting Gamma(r) for every temperature ---
    print("\nFitting Gamma(r) for all temperatures to find correlation length...")
    xi_from_fit_array = []
    fit_start, fit_end = 2, L // 4 # Define a stable range to fit over

    for result in sorted_results:
        Gamma_r = np.array(result['correlation_function'])
        r_values = np.arange(len(Gamma_r))
        
        valid_indices = np.where(Gamma_r > 1e-9)[0]
        fit_indices = np.intersect1d(np.arange(fit_start, fit_end + 1), valid_indices)

        if len(fit_indices) > 2:
            try:
                r_fit = r_values[fit_indices]
                log_Gamma_fit = np.log(Gamma_r[fit_indices])
                slope, _, _, _, _ = linregress(r_fit, log_Gamma_fit)
                
                # Check for a valid negative slope
                if slope < 0:
                    xi_from_fit_array.append(-1 / slope)
                else:
                    xi_from_fit_array.append(np.nan) # Invalid fit
            except Exception:
                xi_from_fit_array.append(np.nan) # Fit failed
        else:
            xi_from_fit_array.append(np.nan) # Not enough data to fit

    correlation_length_fit_array = np.array(xi_from_fit_array)
    print("Fitting complete.")

    # --- UPDATED: Save the new correlation length to the results file ---
    results_file = os.path.join(output_dir, f"L_{L}_results.txt")
    header = "# Temperature Magnetization Energy Binder Susceptibility SpecificHeat CorrLength(Fit)"
    data_to_save = np.vstack([
        temps_array, mag_array, energy_array, binder_array, 
        susceptibility_array, specific_heat_array, correlation_length_fit_array
    ]).T
    np.savetxt(results_file, data_to_save, header=header, fmt='%.8f')
    print(f"Saved aggregated results to {results_file}")

    # --- PART 1: Plotting of Bulk Quantities ---
    print("Generating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Ising Model Physical Properties (L={L})', fontsize=16)

    # (Plots for M, E, Chi, Cv, Binder remain the same)
    axes[0, 0].plot(temps_array, mag_array, 'o-', markersize=4)
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
    axes[2, 1].plot(temps_array, correlation_length_fit_array, '*-', color='brown')
    axes[2, 1].set_ylabel('Correlation Length $\\xi$ (from Γ(r) fit)')
    axes[2, 1].set_xlabel('Temperature $T$')
    axes[2, 1].grid(True)
    
    Tc_exact = 2 / np.log(1 + np.sqrt(2))
    for ax_row in axes:
        for ax in ax_row:
            ax.axvline(x=Tc_exact, color='gray', linestyle='--', linewidth=2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_file = os.path.join(output_dir, f"L_{L}_bulk_plots.png")
    plt.savefig(plot_file)
    print(f"Saved bulk property plots to {plot_file}")
    plt.close(fig)