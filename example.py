from build import pyising

def main():
    L = 16
    seed = 958431198
    N = 100000
    tstar = 2.4

    model = pyising.Ising2D(L, seed)
    model.initialize_spins()
    model.compute_neighbors()

    # We must call compute_energy() at least once to properly set internal energy
    model.compute_energy()

    print("Running Metropolis step...")
    model.do_step_metropolis(tstar, N)
    print(f"Mean Magnetization = {model.get_magnetization()}")
    print(f"Mean Energy       = {model.get_energy_mean()}")
    print(f"Binder Cumulant   = {model.get_binder_cumulant()}")

    print("\nResetting spins and running Wolff step...")
    model.initialize_spins()
    model.compute_neighbors()
    model.compute_energy()

    model.do_step_wolff(tstar, N)
    print(f"Mean Magnetization = {model.get_magnetization()}")
    print(f"Mean Energy       = {model.get_energy_mean()}")
    print(f"Binder Cumulant   = {model.get_binder_cumulant()}")

if __name__ == "__main__":
    main()