import os
from pyising import pyising
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_ising(lattice_sizes, seed, steps, temperatures, output_dir="simulation_data"):
    """
    Simulate the 2D Ising model over a range of lattice sizes and temperatures, storing results.

    Parameters:
    - lattice_sizes (list of int): List of lattice sizes (e.g., [16, 32, 64]).
    - seed (int): Seed for random number generator.
    - steps (int): Number of simulation steps per temperature.
    - temperatures (list or np.array): List of temperatures to simulate.
    - output_dir (str): Directory to store simulation data.

    Returns:
    - results (dict): Nested dictionary containing averaged properties for each lattice size and temperature.
    """
    # Initialize a nested dictionary to store results
    results = {}

    # Ensure the main output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created main output directory: {output_dir}")

    # Loop over each lattice size
    for L in tqdm(lattice_sizes, desc="Simulating Lattice Sizes"):
        results[L] = {}  # Initialize dictionary for this lattice size

        # Format lattice size for directory naming
        L_formatted = f"L_{L}"
        size_dir = os.path.join(output_dir, L_formatted)

        # Create a subdirectory for the current lattice size
        if not os.path.exists(size_dir):
            os.makedirs(size_dir)
            print(f"  Created directory for lattice size {L}: {size_dir}")

        # Loop over each temperature
        for T in tqdm(temperatures, desc=f"Simulating Temperatures for L={L}", leave=False):
            # Format temperature to two decimal places for directory naming
            T_formatted = f"T_{T:.2f}"
            temp_dir = os.path.join(size_dir, T_formatted)

            # Create a subdirectory for the current temperature
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                print(f"    Created directory for T={T_formatted}: {temp_dir}")

            # Initialize the Ising model
            model = pyising.Ising2D(L, seed)
            model.initialize_spins()
            model.compute_neighbors()  # Ensure neighbors are computed

            # Thermalize the system before measurements
            model.do_step_metropolis(T, 1000)  # Thermalization steps

            # Initialize accumulators for measurements
            mag_sum = 0.0
            ene_sum = 0.0
            mag2_sum = 0.0
            ene2_sum = 0.0
            mag4_sum = 0.0

            # List to store spin configurations
            configurations = []

            # Perform simulation steps
            for step in range(1, steps + 1):
                model.do_step_metropolis(T, 1)  # Single Metropolis step

                # Retrieve measurements
                mag = model.get_magnetization()
                ene = model.get_energy_mean()

                mag_sum += mag
                ene_sum += ene
                mag2_sum += model.get_magnetization2()
                ene2_sum += model.get_energy2()
                mag4_sum += model.get_magnetization4()

                # Retrieve the current spin configuration
                config_list = model.get_configuration()
                configurations.append(config_list)

                # Optional: Save configurations at specific intervals to manage memory
                # For example, save every 100 steps
                # if step % 100 == 0:
                #     config_array = np.array(configurations)
                #     np.save(os.path.join(temp_dir, f"config_step_{step}.npy"), config_array)
                #     configurations = []

            # Calculate averages
            avg_mag = mag_sum / steps
            avg_ene = ene_sum / steps
            avg_mag2 = mag2_sum / steps
            avg_ene2 = ene2_sum / steps
            avg_mag4 = mag4_sum / steps

            # Total number of spins
            N = L * L

            # Calculate Binder cumulant: U = 1 - <m^4> / (3 <m^2>^2)
            binder = 1.0 - (avg_mag4 / (3.0 * (avg_mag2 ** 2)))

            # Calculate Magnetic Susceptibility: χ = N ( <m²> - <m>² ) / T
            susceptibility = (avg_mag2 - avg_mag**2) * N / T

            # Calculate Specific Heat: C = ( <E²> - <E>² ) / (T²)
            specific_heat = (avg_ene2 - avg_ene**2) / (T ** 2)

            # Store the results in the nested dictionary
            results[L][T] = {
                'Magnetization': avg_mag,
                'Energy': avg_ene,
                'Binder Cumulant': binder,
                'Magnetic Susceptibility': susceptibility,
                'Specific Heat': specific_heat
            }

            # Convert configurations list to a NumPy array for efficient storage
            configurations_np = np.array(configurations, dtype=np.int32)

            # Define the filename for storing configurations
            config_filename = os.path.join(temp_dir, "configurations.npy")

            # Save all configurations for the current temperature and lattice size
            np.save(config_filename, configurations_np)
            print(f"    Saved configurations to {config_filename}")

        # Optional: Save aggregated results for this lattice size to a CSV file
        save_results_csv(results, L, size_dir)

    return results

def save_results_csv(results, L, size_dir):
    """
    Save the aggregated results for a specific lattice size to a CSV file.

    Parameters:
    - results (dict): Nested dictionary containing simulation results.
    - L (int): Current lattice size.
    - size_dir (str): Directory path for the current lattice size.
    """
    # Extract all temperatures for this lattice size
    temperatures = sorted(results[L].keys())

    # Prepare data for CSV
    data = []
    for T in temperatures:
        data.append([
            T,
            results[L][T]['Magnetization'],
            results[L][T]['Energy'],
            results[L][T]['Binder Cumulant'],
            results[L][T]['Magnetic Susceptibility'],
            results[L][T]['Specific Heat']
        ])

    # Convert to NumPy array
    data_array = np.array(data)

    # Define the path for the results file
    results_filepath = os.path.join(size_dir, "results.csv")

    # Save the results to CSV
    header = "Temperature,Magnetization,Energy,Binder_Cumulant,Magnetic_Susceptibility,Specific_Heat"
    np.savetxt(results_filepath, data_array, delimiter=",", header=header, comments='', fmt=['%.2f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f'])
    print(f"  Saved aggregated results to {results_filepath}")

def plot_results(results, lattice_sizes, temperatures_to_plot=None):
    """
    Plot the simulation results for specified lattice sizes.

    Parameters:
    - results (dict): Nested dictionary containing simulation results.
    - lattice_sizes (list of int): List of lattice sizes to plot.
    - temperatures_to_plot (list of float, optional): Specific temperatures to highlight in the plots.
    """
    for L in lattice_sizes:
        plt.figure(figsize=(18, 10))
        plt.suptitle(f"Simulation Results for Lattice Size L={L}", fontsize=16)

        temperatures = sorted(results[L].keys())
        magnetizations = [results[L][T]['Magnetization'] for T in temperatures]
        energies = [results[L][T]['Energy'] for T in temperatures]
        binder_cumulants = [results[L][T]['Binder Cumulant'] for T in temperatures]
        susceptibilities = [results[L][T]['Magnetic Susceptibility'] for T in temperatures]
        specific_heats = [results[L][T]['Specific Heat'] for T in temperatures]

        # Plot Magnetization vs Temperature
        plt.subplot(2, 3, 1)
        plt.plot(temperatures, magnetizations, 'o-', color='blue')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Average Magnetization ⟨m⟩')
        plt.title(f'Magnetization vs Temperature for L={L}')
        plt.grid(True)

        # Plot Energy vs Temperature
        plt.subplot(2, 3, 2)
        plt.plot(temperatures, energies, 's-', color='red')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Average Energy ⟨E⟩')
        plt.title(f'Energy vs Temperature for L={L}')
        plt.grid(True)

        # Plot Binder Cumulant vs Temperature
        plt.subplot(2, 3, 3)
        plt.plot(temperatures, binder_cumulants, 'd-', color='green')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Binder Cumulant U')
        plt.title(f'Binder Cumulant vs Temperature for L={L}')
        plt.grid(True)

        # Plot Magnetic Susceptibility vs Temperature
        plt.subplot(2, 3, 4)
        plt.plot(temperatures, susceptibilities, '^-', color='purple')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Magnetic Susceptibility χ')
        plt.title(f'Magnetic Susceptibility vs Temperature for L={L}')
        plt.grid(True)

        # Plot Specific Heat vs Temperature
        plt.subplot(2, 3, 5)
        plt.plot(temperatures, specific_heats, 'v-', color='orange')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Specific Heat C')
        plt.title(f'Specific Heat vs Temperature for L={L}')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def main():
    # Simulation parameters
    lattice_sizes = [16, 32, 64]          # List of lattice sizes to simulate
    seed = 45646                          # Random seed for reproducibility
    steps = 1000                          # Number of simulation steps per temperature
    temperatures = np.linspace(1.0, 4.0, 30)  # Temperature range from 1.0 to 4.0 with 30 points
    output_dir = "simulation_data"        # Main directory to store simulation data

    # Run simulations
    results = simulate_ising(lattice_sizes, seed, steps, temperatures, output_dir=output_dir)

    # Plot the results for each lattice size
    plot_results(results, lattice_sizes)

if __name__ == "__main__":
    main()