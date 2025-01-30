#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <map>

struct Results {
    // Existing members
    double binder;
    double meanMag;
    double meanMag2;
    double meanMag4;
    double meanEne;
    double meanEne2;
    double meanEne4;
    double T;
    int L;

    std::vector<int> configuration;
    std::vector<std::vector<int>> all_configurations;

    std::map<std::string, std::string> metadata;  // Store parameters, versions, etc
    std::chrono::duration<double> runtime;       // Execution time
    std::vector<double> timing_per_step;         // Per-step timing

};
    std::vector<Results> run_parallel_metropolis(
        const std::vector<double>& temps, int L, int N_steps,
        unsigned int seed_base, const std::string& output_dir,
        bool use_wolff, bool save_all_configs
        );
class Ising2D
{
public:
    Ising2D(int L, unsigned int seed);

    bool m_save_all_configs = false;
    std::vector<std::vector<int>> m_all_configs;
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
        Results res;
        // Copy over the observables to the struct
        res.binder          = m_binder;
        res.meanMag         = m_meanMag;
        res.meanMag2        = m_meanMag2;
        res.meanMag4        = m_meanMag4;
        res.meanEne         = m_meanEne;
        res.meanEne2        = m_meanEne2;
        res.meanEne4        = m_meanEne4;
        // Spins/configurations
        res.configuration   = get_configuration();
        res.all_configurations = m_all_configs;
        return res;
    }

    void enable_save_all_configs(bool enable) {
        m_save_all_configs = enable;
    }
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