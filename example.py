import os
from build import pyising
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_ising(L, seed, steps, temperatures, output_dir="simulation_data"):
    """
    Simulate the 2D Ising model over a range of temperatures and store results.

    Parameters:
    - L (int): Lattice size (L x L).
    - seed (int): Seed for random number generator.
    - steps (int): Number of simulation steps per temperature.
    - temperatures (list or np.array): List of temperatures to simulate.
    - output_dir (str): Directory to store simulation data.

    Returns:
    - results (dict): Dictionary containing averaged properties for each temperature.
    """
    # Total number of spins
    N = L * L

    # Initialize lists to store results
    results = {
        'Temperature': [],
        'Magnetization': [],
        'Energy': [],
        'Binder Cumulant': [],
        'Magnetic Susceptibility': [],
        'Specific Heat': []
    }

    # Ensure the main output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Loop over each temperature
    for T in tqdm(temperatures, desc="Simulating Temperatures"):
        # Format temperature to two decimal places for directory naming
        T_formatted = f"{T:.2f}"
        temp_dir = os.path.join(output_dir, f"T_{T_formatted}")

        # Create a subdirectory for the current temperature
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            print(f"  Created directory for T={T_formatted}: {temp_dir}")

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

        # Calculate Binder cumulant: U = 1 - <m^4> / (3 <m^2>^2)
        binder = 1.0 - (avg_mag4 / (3.0 * (avg_mag2 ** 2)))

        # Calculate Magnetic Susceptibility: χ = N ( <m²> - <m>² ) / T
        susceptibility = (avg_mag2 - avg_mag**2) * N / T

        # Calculate Specific Heat: C = ( <E²> - <E>² ) / (T²)
        specific_heat = (avg_ene2 - avg_ene**2) / (T ** 2)

        # Store the results
        results['Temperature'].append(T)
        results['Magnetization'].append(avg_mag)
        results['Energy'].append(avg_ene)
        results['Binder Cumulant'].append(binder)
        results['Magnetic Susceptibility'].append(susceptibility)
        results['Specific Heat'].append(specific_heat)

        # Convert configurations list to a NumPy array for efficient storage
        configurations_np = np.array(configurations, dtype=np.int32)

        # Define the filename for storing configurations
        config_filename = os.path.join(temp_dir, "configurations.npy")

        # Save all configurations for the current temperature
        np.save(config_filename, configurations_np)
        print(f"  Saved configurations to {config_filename}")

    # Optionally, save the aggregated results to a CSV file for easy analysis
    results_array = np.array([
        results['Temperature'],
        results['Magnetization'],
        results['Energy'],
        results['Binder Cumulant'],
        results['Magnetic Susceptibility'],
        results['Specific Heat']
    ]).T  # Transpose to have each row correspond to a temperature

    # Define the path for the results file
    results_filepath = os.path.join(output_dir, "results.csv")

    # Save the results to CSV
    np.savetxt(results_filepath, results_array, delimiter=",",
               header="Temperature,Magnetization,Energy,Binder_Cumulant,Magnetic_Susceptibility,Specific_Heat",
               comments='')
    print(f"Saved aggregated results to {results_filepath}")

    return results

def plot_results(results):
    """
    Plot the simulation results.

    Parameters:
    - results (dict): Dictionary containing averaged properties for each temperature.
    """
    temperatures = results['Temperature']
    magnetizations = results['Magnetization']
    energies = results['Energy']
    binder_cumulants = results['Binder Cumulant']
    susceptibilities = results['Magnetic Susceptibility']
    specific_heats = results['Specific Heat']

    plt.figure(figsize=(18, 10))

    # Plot Magnetization vs Temperature
    plt.subplot(2, 3, 1)
    plt.plot(temperatures, magnetizations, 'o-', color='blue')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Magnetization ⟨m⟩')
    plt.title('Magnetization vs Temperature')
    plt.grid(True)

    # Plot Energy vs Temperature
    plt.subplot(2, 3, 2)
    plt.plot(temperatures, energies, 's-', color='red')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Energy ⟨E⟩')
    plt.title('Energy vs Temperature')
    plt.grid(True)

    # Plot Binder Cumulant vs Temperature
    plt.subplot(2, 3, 3)
    plt.plot(temperatures, binder_cumulants, 'd-', color='green')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Binder Cumulant U')
    plt.title('Binder Cumulant vs Temperature')
    plt.grid(True)

    # Plot Magnetic Susceptibility vs Temperature
    plt.subplot(2, 3, 4)
    plt.plot(temperatures, susceptibilities, '^-', color='purple')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Magnetic Susceptibility χ')
    plt.title('Magnetic Susceptibility vs Temperature')
    plt.grid(True)

    # Plot Specific Heat vs Temperature
    plt.subplot(2, 3, 5)
    plt.plot(temperatures, specific_heats, 'v-', color='orange')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Specific Heat C')
    plt.title('Specific Heat vs Temperature')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Simulation parameters
    L = 16                  # Lattice size (16x16)
    seed = 45646            # Random seed for reproducibility
    steps = 1000            # Number of simulation steps per temperature
    temperatures = np.linspace(1.0, 4.0, 30)  # Temperature range from 1.0 to 4.0 with 30 points
    output_dir = "simulation_data"            # Main directory to store simulation data

    # Run simulations
    results = simulate_ising(L, seed, steps, temperatures, output_dir=output_dir)

    # Plot the results
    plot_results(results)

if __name__ == "__main__":
    main()