#include <iostream>
#include "ising.hpp"

int main()
{
    int L = 16;
    unsigned int seed = 958431198;
    int N = 100000;
    double tstar = 2.4;

    Ising2D model(L, seed);
    model.initialize_spins();
    model.compute_neighbors();
    model.compute_energy(); // sets internal energy

    // Metropolis
    model.do_step_metropolis(tstar, N);
    std::cout << "Metropolis results:\n";
    std::cout << "Mean Magnetization = " << model.get_magnetization() << "\n";
    std::cout << "Mean Energy       = " << model.get_energy_mean() << "\n";
    std::cout << "Binder Cumulant   = " << model.get_binder_cumulant() << "\n";

    // Wolff
    model.initialize_spins();
    model.compute_neighbors();
    model.compute_energy(); // reset internal energy
    model.do_step_wolff(tstar, N);
    std::cout << "\nWolff results:\n";
    std::cout << "Mean Magnetization = " << model.get_magnetization() << "\n";
    std::cout << "Mean Energy       = " << model.get_energy_mean() << "\n";
    std::cout << "Binder Cumulant   = " << model.get_binder_cumulant() << "\n";

    return 0;
}