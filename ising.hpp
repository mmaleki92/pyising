#pragma once

#include <vector>
#include <random>

struct Results {
    double binder;
    double meanMag;
    double meanEne;
};

class Ising2D
{
public:
    Ising2D(int L, unsigned int seed);
    // Public API
    void initialize_spins();
    void compute_neighbors();
    double compute_energy();
    double magnetization() const;
    // Existing "batch" methods (optional to keep)
    void do_step_metropolis(double tstar, int N);
    void do_step_wolff(double tstar, int N);

    // NEW addition: single-step methods for Python loops
    void do_metropolis_step(double tstar);
    void do_wolff_step(double tstar);
    
    // Method to get the current spin configuration as +1/-1
    std::vector<int> get_configuration() const;

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
    Results get_results() const {
        return {m_binder, m_meanMag, m_meanEne};
    };
private:
    // Internal methods
    void metropolis_flip_spin();
    void wolff_add_to_cluster(int pos, double p);
    void measure_observables(double N);
    void thermalize_metropolis(double tstar);
    void thermalize_wolff(double tstar);
    int m_L;
    int m_SIZE;

    std::mt19937 m_gen;
    std::uniform_int_distribution<int> m_ran_pos; 
    std::uniform_real_distribution<double> m_ran_u;
    std::uniform_int_distribution<int> m_brandom;
    
    std::vector<bool> m_spins;
    std::vector< std::vector<int> > m_neighbors;
    std::vector<double> m_h; // Changed from double m_h[5] to std::vector<double>
    double m_energy;

    // Measured quantities
    double m_meanMag;
    double m_meanMag2;
    double m_meanMag4;
    double m_meanEne;
    double m_meanEne2;
    double m_meanEne4;
    double m_binder;

private:
    // Private methods for flipping spins, building clusters, etc.
    void compute_metropolis_factors(double tstar);

    // NEW utility method: interpret bool spin as ±1
    inline int spin_val(bool s) const { return (s ? +1 : -1); }
};