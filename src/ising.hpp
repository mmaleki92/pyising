#pragma once
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <pcg_random.hpp>
#include <cmath>
#include <mpi.h>


struct Results {
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
    std::map<std::string, std::string> metadata;
    std::chrono::duration<double> runtime;
    std::vector<double> timing_per_step;
};

// This is the pure C++ simulation function. It knows nothing about Python.
std::vector<Results> run_simulation_cpp(
    const std::vector<double>& temps, int L, int N_steps,
    int equ_N, int snapshot_interval, unsigned int seed_base,
    const std::string& output_dir, bool use_wolff,
    bool save_all_configs);

class Ising2D 
{
public:
    Ising2D(int L, unsigned int seed);
    void initialize_spins();
    void compute_neighbors();
    void do_step_metropolis(double tstar, int N, int equ_N, int snapshot_interval);
    void do_step_wolff(double tstar, int N, int snapshot_interval);
    double compute_energy();
    double magnetization() const;
    void do_metropolis_step(double tstar);
    void do_wolff_step(double tstar);
    std::vector<int> get_configuration() const;
    double get_magnetization() const { return m_meanMag; }
    double get_magnetization2() const { return m_meanMag2; }
    double get_magnetization4() const { return m_meanMag4; }
    double get_energy_mean() const { return m_meanEne; }
    double get_energy2() const { return m_meanEne2; }
    double get_energy4() const { return m_meanEne4; }
    double get_binder_cumulant() const { return m_binder; }
    int get_L() const { return m_L; }
    Results get_results() const;
    void enable_save_all_configs(bool enable);
    void set_config_save_path(const std::string& path);
    void set_snapshot_interval(int interval);
    void save_current_config(int step_number);

private:
    // Internal methods
    // void wolff_add_to_cluster(int pos, double p);


    // RNG
    pcg32 m_gen;
    std::uniform_int_distribution<int> m_ran_pos;
    std::uniform_real_distribution<double> m_ran_u;
    std::vector<char> m_spins;
    std::vector<int> m_neighbors;
    int m_L, m_SIZE;
    double m_energy;
    std::vector<double> m_h;
    double m_meanMag, m_meanMag2, m_meanMag4;
    double m_meanEne, m_meanEne2, m_meanEne4;
    double m_binder;
    bool m_save_all_configs;
    int m_snapshot_count;
    int m_snapshot_interval;
    std::string m_config_save_path;
    inline int wrap(int coord) const { return (coord + m_L) % m_L; }
    void metropolis_flip_spin(double tstar);
    void wolff_cluster_update(double p);
    void compute_metropolis_factors(double tstar);
    void thermalize_wolff(double tstar);
};