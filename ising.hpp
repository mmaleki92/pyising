#pragma once

#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>

class Ising2D
{
public:
    Ising2D(int L, unsigned int seed);

    // Public API
    void initialize_spins();
    void compute_neighbors();
    double compute_energy();
    double magnetization() const;
    void do_step_metropolis(double tstar, int N);
    void do_step_wolff(double tstar, int N);

    // Accessors for measured quantities
    double get_magnetization() const { return m_meanMag; }
    double get_magnetization2() const { return m_meanMag2; }
    double get_magnetization4() const { return m_meanMag4; }
    double get_energy_mean() const { return m_meanEne; }
    double get_energy2() const { return m_meanEne2; }
    double get_energy4() const { return m_meanEne4; }
    double get_binder_cumulant() const { return m_binder; }

    // Number of spins in one lattice dimension
    int get_L() const { return m_L; }

private:
    // Internal methods
    void metropolis_flip_spin();
    void wolff_add_to_cluster(int pos, double p);
    void measure_observables(double N);
    void thermalize_metropolis(double tstar);
    void thermalize_wolff(double tstar);

private:
    int m_L;
    int m_SIZE;
    std::mt19937 m_gen;
    std::uniform_int_distribution<int> m_ran_pos; 
    std::uniform_real_distribution<double> m_ran_u;
    std::uniform_int_distribution<int> m_brandom;
    
    // Spin storage
    // Using bool to store spin up/down, internally interpret: 
    // spin[i] = 0 => -1, spin[i] = 1 => +1
    std::vector<bool> m_spins;

    // Neighbors: [site_index][0..3]
    std::vector< std::vector<int> > m_neighbors;

    // Precomputed exponential factors for Metropolis acceptance
    double m_h[5]; // h indices: h[(deltaE+4)/2] due to i in [-4, -2, 0, 2, 4]

    // Current energy
    double m_energy;

    // Measured quantities
    double m_meanMag;
    double m_meanMag2;
    double m_meanMag4;
    double m_meanEne;
    double m_meanEne2;
    double m_meanEne4;
    double m_binder;

    // Helpers
    void compute_metropolis_factors(double tstar);
    inline int spin_val(bool s) const { return (s ? 1 : -1); }
};