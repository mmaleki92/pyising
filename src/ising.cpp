#include "ising.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include "cnpy/cnpy.h"
#include <omp.h>
#include <indicators/progress_bar.hpp>
#include <indicators/termcolor.hpp>


#define UP    0
#define RIGHT 1
#define LEFT  2
#define DOWN  3

std::vector<Results> run_parallel_metropolis(
    const std::vector<double>& temps,
    int L,
    int N_steps,
    unsigned int seed_base,
    const std::string& output_dir,
    bool use_wolff,
    bool save_all_configs
) {
    std::vector<Results> results(temps.size());

    // Create directory for current L
    std::string L_dir = output_dir + "/L_" + std::to_string(L);
    #pragma omp critical
    {
        std::filesystem::create_directories(L_dir);
    }

    // Configure the progress bar
    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PostfixText{"Running simulations..."},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::MaxProgress{temps.size()},
        // Hide the bar from the console once it's marked as completed
    };

    // Shared counter for progress
    std::atomic<size_t> progress_counter{0};

    // Parallel loop
    #pragma omp parallel for
    for (size_t i = 0; i < temps.size(); ++i) {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        unsigned int seed = seed_base + static_cast<unsigned int>(i);
        Ising2D model(L, seed);
        model.initialize_spins();
        model.compute_neighbors();

        if (save_all_configs) {
            model.enable_save_all_configs(true);
        }

        // Decide Metropolis vs Wolff
        if (use_wolff) {
            model.do_step_wolff(temps[i], N_steps);
        } else {
            model.do_step_metropolis(temps[i], N_steps);
        }

        results[i] = model.get_results();
        results[i].T = temps[i];
        results[i].L = L;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << temps[i];
        std::string T_str = ss.str();
        std::string T_dir;

        // Create a subdirectory for each temperature
        #pragma omp critical
        {
            T_dir = L_dir + "/T_" + T_str;
            std::filesystem::create_directories(T_dir);
        }

        // Save configurations
        if (save_all_configs) {
            std::string all_filename = T_dir + "/all_configs.npy";
            const auto& all_configs = results[i].all_configurations;
            if (!all_configs.empty()) {
                size_t num_steps = all_configs.size();
                size_t L_size = static_cast<size_t>(L);

                std::vector<int> flattened;
                flattened.reserve(num_steps * L_size * L_size);
                for (const auto& config : all_configs) {
                    flattened.insert(flattened.end(), config.begin(), config.end());
                }
                #pragma omp critical
                {
                    cnpy::npy_save(all_filename, flattened.data(), {num_steps, L_size, L_size}, "w");
                }
            }
        } else {
            // Save only the final configuration
            std::string filename = T_dir + "/config.npy";
            const std::vector<int>& config = results[i].configuration;
            cnpy::npy_save(filename, config.data(), {static_cast<size_t>(L), static_cast<size_t>(L)}, "w");
        }


        // Update the progress bar
        size_t current_count = ++progress_counter;
        #pragma omp critical
        {
            bar.set_progress(current_count);
        }
    }

    // Mark the bar as complete when done
    // bar.mark_as_completed();

    // Print a final message
    // std::cout << "All simulations completed successfully!\n";

    return results;
}
void Ising2D::do_metropolis_step(double tstar)
{

    compute_metropolis_factors(tstar);

    // Perform exactly one spin-flip attempt
    metropolis_flip_spin();
}

// NEW addition: single-step method for Wolff
void Ising2D::do_wolff_step(double tstar)
{
    // Probability for adding a neighbor to the cluster
    double p = 1.0 - std::exp(-2.0 / tstar);

    // Pick a random spin
    int pos = m_ran_pos(m_gen);

    // Flip cluster around pos
    wolff_add_to_cluster(pos, p);
}

// Existing method (for reference)
std::vector<int> Ising2D::get_configuration() const
{
    std::vector<int> config(m_SIZE, 0);
    for (int i = 0; i < m_SIZE; i++)
    {
        config[i] = spin_val(m_spins[i]);  // +1 or -1
    }
    return config;
}

Ising2D::Ising2D(int L, unsigned int seed)
    : m_L(L),
      m_SIZE(L*L),
      m_gen(seed),
      m_ran_pos(0, L*L - 1),
      m_ran_u(0.0, 1.0),
      m_brandom(0, 1),
      m_spins(L*L, false),
      m_neighbors(L*L, std::vector<int>(4, 0)),
      m_energy(0.0),
      m_meanMag(0.0),
      m_meanMag2(0.0),
      m_meanMag4(0.0),
      m_meanEne(0.0),
      m_meanEne2(0.0),
      m_meanEne4(0.0),
      m_binder(0.0)
{
}

void Ising2D::initialize_spins()
{
    for (int i = 0; i < m_SIZE; i++) {
        m_spins[i] = (m_brandom(m_gen) == 1);
    }
    m_energy = compute_energy();  // Initialize energy correctly
}


void Ising2D::compute_neighbors()
{
    for (int i = 0; i < m_L; i++) {
        for (int j = 0; j < m_L; j++) {
            int idx = i + j*m_L;
            int u = (j+1 == m_L) ? 0 : j+1;
            int d = (j-1 < 0)    ? m_L-1 : j-1;
            int r = (i+1 == m_L) ? 0 : i+1;
            int l = (i-1 < 0)    ? m_L-1 : i-1;

            m_neighbors[idx][UP]    = i + u*m_L;
            m_neighbors[idx][DOWN]  = i + d*m_L;
            m_neighbors[idx][RIGHT] = r + j*m_L;
            m_neighbors[idx][LEFT]  = l + j*m_L;
        }
    }
}

double Ising2D::compute_energy()
{
    int totalEnergy = 0;
    for (int i = 0; i < m_SIZE; i++) {
        int s_i = spin_val(m_spins[i]);
        int sum_neigh = spin_val(m_spins[m_neighbors[i][UP]])
                      + spin_val(m_spins[m_neighbors[i][DOWN]])
                      + spin_val(m_spins[m_neighbors[i][RIGHT]])
                      + spin_val(m_spins[m_neighbors[i][LEFT]]);
        totalEnergy += s_i * sum_neigh;
    }
    return -totalEnergy / 2.0;  // Correct energy calculation
}


double Ising2D::magnetization() const
{
    double sum = 0.0;
    for (int i = 0; i < m_SIZE; i++) {
        sum += spin_val(m_spins[i]);
    }
    return (sum / (double)m_SIZE);
}

void Ising2D::compute_metropolis_factors(double tstar) {
    m_h.resize(5);  // For deltaE values: -8, -4, 0, +4, +8
    for (int deltaE : {-8, -4, 0, 4, 8}) {
        int idx = (deltaE + 8) / 4;
        if (deltaE <= 0) {
            m_h[idx] = 1.0;
        } else {
            m_h[idx] = std::exp(-deltaE / tstar);
        }
    }
}

void Ising2D::metropolis_flip_spin()
{
    int index = m_ran_pos(m_gen);
    int sum_neigh = spin_val(m_spins[m_neighbors[index][UP]])
                  + spin_val(m_spins[m_neighbors[index][DOWN]])
                  + spin_val(m_spins[m_neighbors[index][RIGHT]])
                  + spin_val(m_spins[m_neighbors[index][LEFT]]);
    int currentSpin = spin_val(m_spins[index]);
    int deltaE = 2 * currentSpin * sum_neigh;

    int idx = (deltaE + 8) / 4;
    if (idx < 0 || idx >= m_h.size()) {
        // Handle error: unexpected deltaE value
        return;
    }

    if (m_ran_u(m_gen) < m_h[idx]) {
        m_spins[index] = !m_spins[index];
        m_energy += deltaE;  // Update total energy correctly
    }
}

void Ising2D::thermalize_metropolis(double tstar)
{
    // Example: 1100 steps was used in original code
    // This is arbitrary, you can adjust as needed
    for (int i = 0; i < 1100; i++) {
        metropolis_flip_spin();
    }
}

void Ising2D::measure_observables(double N) {
    double mag    = std::fabs(magnetization());
    double mag2   = mag * mag;
    double ene    = m_energy;
    double ene2   = ene * ene;

    // Accumulate
    m_meanMag  += mag;
    m_meanMag2 += mag2;
    m_meanMag4 += (mag2 * mag2);
    m_meanEne  += ene;
    m_meanEne2 += ene2;
    m_meanEne4 += (ene2 * ene2);
    if (m_save_all_configs) {
            m_all_configs.push_back(get_configuration());
    }
}

void Ising2D::do_step_metropolis(double tstar, int N) {
    // Reset accumulators
    m_meanMag  = 0.0;
    m_meanMag2 = 0.0;
    m_meanMag4 = 0.0;
    m_meanEne  = 0.0;
    m_meanEne2 = 0.0;
    m_meanEne4 = 0.0;
    m_binder   = 0.0;

    // Precompute the acceptance factors
    compute_metropolis_factors(tstar);

    // Thermalize
    thermalize_metropolis(tstar);

    // Perform N iterations each followed by partial "equilibration"
    // (like original code: 1100 flips per measurement)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 1100; j++)
            metropolis_flip_spin();

        // Measure
        measure_observables((double)N);
    }

    // Convert from sum to average
    m_meanMag  /= (double)N;
    m_meanMag2 /= (double)N;
    m_meanMag4 /= (double)N;
    m_meanEne  /= (double)N;
    m_meanEne2 /= (double)N;
    m_meanEne4 /= (double)N;

    // Binder cumulant:
    // U = 1 - <m^4> / (3 <m^2>^2)
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
}

void Ising2D::thermalize_wolff(double tstar) {
    // For example ~15 cluster updates, from original code
    for (int i = 0; i < 15; i++) {
        int pos = m_ran_pos(m_gen);
        double p = 1.0 - std::exp(-2.0 / tstar);
        wolff_add_to_cluster(pos, p);
    }
}

void Ising2D::wolff_add_to_cluster(int pos, double p) {
    // Calculate local energy contribution first
    int sum_neigh = spin_val(m_spins[m_neighbors[pos][UP]])
                  + spin_val(m_spins[m_neighbors[pos][DOWN]])
                  + spin_val(m_spins[m_neighbors[pos][RIGHT]])
                  + spin_val(m_spins[m_neighbors[pos][LEFT]]);

    // if spin=+1 => deltaE = 2*sum_neigh -4
    // if spin=-1 => deltaE = 4 -2*sum_neigh
    int deltaE = 0;
    if (m_spins[pos])
        deltaE = (2 * sum_neigh - 4);
    else
        deltaE = (4 - 2 * sum_neigh);

    // Flip the spin, adjust energy
    m_energy += (2.0 * deltaE) / (m_SIZE * 1.0);
    m_spins[pos] = !m_spins[pos];

    // Now check neighbors
    int newSpinVal = spin_val(m_spins[pos]);
    int oldSpinVal = -newSpinVal; 

    // For each neighbor, if oldSpinVal is found => possibly add to cluster
    for (int i = 0; i < 4; i++) {
        int npos = m_neighbors[pos][i];
        if (spin_val(m_spins[npos]) == oldSpinVal) {
            // Probability p
            if (m_ran_u(m_gen) < p) {
                wolff_add_to_cluster(npos, p);
            }
        }
    }
}

void Ising2D::do_step_wolff(double tstar, int N) {
    m_meanMag  = 0.0;
    m_meanMag2 = 0.0;
    m_meanMag4 = 0.0;
    m_meanEne  = 0.0;
    m_meanEne2 = 0.0;
    m_meanEne4 = 0.0;
    m_binder   = 0.0;

    // Thermalize with some cluster updates
    thermalize_wolff(tstar);

    for (int i = 0; i < N; i++) {
        // Each iteration do ~12 cluster updates
        for (int j = 0; j < 12; j++) {
            int pos = m_ran_pos(m_gen);
            double pa = 1.0 - std::exp(-2.0 / tstar);
            wolff_add_to_cluster(pos, pa);
        }
        measure_observables((double)N);
    }

    m_meanMag  /= (double)N;
    m_meanMag2 /= (double)N;
    m_meanMag4 /= (double)N;
    m_meanEne  /= (double)N;
    m_meanEne2 /= (double)N;
    m_meanEne4 /= (double)N;

    // Binder cumulant
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
}