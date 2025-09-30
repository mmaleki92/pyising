#include <omp.h>
#include <mpi.h>
#include <stack>
#include <string>
#include <iostream>
#include <vector>
#include <filesystem>
#include <sstream>
#include <iomanip>

#include "ising.hpp"
#include "cnpy/cnpy.h"

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
    int equ_N,
    int snapshot_interval,
    unsigned int seed_base,
    const std::string& output_dir,
    bool use_wolff,
    bool save_all_configs
) {
    // DO NOT initialize MPI here. Assume it's already been done by the caller (mpi4py).
    /*
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        // MPI not initialized, initialize it with thread support
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    }
    */

    // Directly get rank and size, assuming a valid MPI environment exists.
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Distribute temperatures among ranks
    int num_temps = static_cast<int>(temps.size());
    // Setup a single progress bar only on rank 0
    indicators::ProgressBar bar;
    if (rank == 0) {
        bar.set_option(indicators::option::BarWidth{50});
        bar.set_option(indicators::option::Start{"["});
        bar.set_option(indicators::option::Fill{"="});
        bar.set_option(indicators::option::Lead{">"});
        bar.set_option(indicators::option::Remainder{" "});
        bar.set_option(indicators::option::End{"]"});
        bar.set_option(indicators::option::PostfixText{"Running simulations..."});
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::yellow});
        bar.set_option(indicators::option::FontStyles{std::vector<indicators::FontStyle>{
            indicators::FontStyle::bold
        }});
        bar.set_option(indicators::option::ShowElapsedTime{true});
        bar.set_option(indicators::option::ShowRemainingTime{true});
        bar.set_option(indicators::option::MaxProgress{
            static_cast<size_t>(num_temps) // One increment per temperature
        });
    }

    int base_local_num = num_temps / size;
    int remainder = num_temps % size;
    int start = rank * base_local_num + std::min(rank, remainder);
    int local_num = base_local_num + (rank < remainder ? 1 : 0);
    int end = start + local_num;

    std::vector<double> local_temps(temps.begin() + start, temps.begin() + end);
    std::vector<Results> local_results(local_temps.size());

    // Create directory for the current L
    std::string L_dir = output_dir + "/L_" + std::to_string(L);
    if (rank == 0) { // Only rank 0 needs to create the main directory
        std::filesystem::create_directories(L_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD); // Ensure directory exists before proceeding

    // Create an MPI window for a global progress counter
    MPI_Win win;
    int* global_counter = nullptr;
    if (rank == 0) {
        MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &global_counter);
        *global_counter = 0;
        MPI_Win_create(global_counter, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(nullptr, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }


    // Parallel loop for each temperature in local_temps
    #pragma omp parallel for
    for (size_t i = 0; i < local_temps.size(); ++i) {
        // Generate a unique seed per temperature
        unsigned int seed = seed_base + static_cast<unsigned int>(start + i);

        // Setup Ising model
        Ising2D model(L, seed);
        model.initialize_spins();
        model.compute_neighbors();
        
        // Format the temperature string for directory naming
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << local_temps[i];
        std::string T_str = ss.str();
        std::string T_dir;

        // Creating the temperature directory
        T_dir = L_dir + "/T_" + T_str;
        // No need for a critical section if each thread/rank works on a unique temperature
        std::filesystem::create_directories(T_dir);

        // If using save_all_configs, tell the model where to save the configs
        if (save_all_configs) {
            model.set_config_save_path(T_dir);
            model.set_snapshot_interval(snapshot_interval);
            model.enable_save_all_configs(true);  // Explicitly enable config saving
        }
        
        // Perform the simulation
        if (use_wolff) {
            model.do_step_wolff(local_temps[i], N_steps, snapshot_interval);
        } else {
            model.do_step_metropolis(local_temps[i], N_steps, equ_N, snapshot_interval);
        }

        // Store and label results - but don't store full configurations in memory
        local_results[i] = model.get_results();
        local_results[i].T = local_temps[i];
        local_results[i].L = L;

        // Save final configuration to a separate file
        std::string filename = T_dir + "/final_config.npy";
        const auto& config = model.get_configuration();  // Get the current configuration
        #pragma omp critical
        {
            cnpy::npy_save(
                filename,
                config.data(),
                {static_cast<size_t>(L), static_cast<size_t>(L)},
                "w"
            );
        }

        // Once this temperature is done, increment the global counter
        int one = 1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);

        // Update the single progress bar (only rank 0, one thread)
        // This check prevents multiple threads on rank 0 from updating the bar concurrently
        if (rank == 0 && omp_get_thread_num() == 0) {
            // A small delay to allow accumulate to complete, though lock/unlock should suffice
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            int current_total = *global_counter;
            MPI_Win_unlock(0, win);
            bar.set_progress(current_total);
        }
    }

    // Synchronize all MPI processes to ensure file I/O is complete
    MPI_Barrier(MPI_COMM_WORLD);

    // Final update to ensure the bar is complete
    if (rank == 0) {
        bar.set_progress(num_temps);
        bar.mark_as_completed();
    }

    // Clean up the MPI window
    MPI_Win_free(&win);
    if (rank == 0 && global_counter != nullptr) {
        MPI_Free_mem(global_counter);
    }

    // You should NOT call MPI_Finalize here. mpi4py will handle it when the script ends.

    return local_results;
}

Results Ising2D::get_results() const {
    Results res;
    res.binder = m_binder;
    res.meanMag = m_meanMag;
    res.meanMag2 = m_meanMag2;
    res.meanMag4 = m_meanMag4;
    res.meanEne = m_meanEne;
    res.meanEne2 = m_meanEne2;
    res.meanEne4 = m_meanEne4;
    res.configuration = get_configuration();  // Only return the current configuration
    res.L = m_L;
    return res;
}

void Ising2D::do_metropolis_step(double tstar)
{
    compute_metropolis_factors(tstar);

    // Perform exactly one spin-flip attempt
    metropolis_flip_spin(tstar);
}


void Ising2D::do_wolff_step(double tstar) {
    // Probability for adding a neighbor to the cluster
    double p = 1.0 - std::exp(-2.0 / tstar);

    // Pick a random spin
    int pos = m_ran_pos(m_gen);

    // Flip cluster around pos
    wolff_cluster_update(p);
}

// Existing method (for reference)
std::vector<int> Ising2D::get_configuration() const {
    return std::vector<int>(m_spins.begin(), m_spins.end());
}


Ising2D::Ising2D(int L, unsigned int seed)
    : m_L(L), m_SIZE(L*L),
      m_gen(seed), m_ran_pos(0, L*L-1), m_ran_u(0.0, 1.0),
      m_spins(L*L, 1), m_neighbors(4*L*L, 0),
      m_energy(0.0), m_save_all_configs(false),
      m_snapshot_count(0), m_snapshot_interval(100),
      m_config_save_path("") {}


void Ising2D::initialize_spins() {
    std::uniform_int_distribution<int> init_dist(0, 1);
    for (auto& s : m_spins) {
        s = init_dist(m_gen) ? 1 : -1;
    }
    compute_neighbors();
    m_energy = compute_energy();
    m_snapshot_count = 0;
}


void Ising2D::compute_neighbors() {
    for (int y = 0; y < m_L; ++y) {
        for (int x = 0; x < m_L; ++x) {
            const int idx = y * m_L + x;
            m_neighbors[4*idx]     = y * m_L + wrap(x + 1);    // Right
            m_neighbors[4*idx + 1] = y * m_L + wrap(x - 1);    // Left
            m_neighbors[4*idx + 2] = wrap(y + 1) * m_L + x;    // Down
            m_neighbors[4*idx + 3] = wrap(y - 1) * m_L + x;    // Up
        }
    }
}

double Ising2D::compute_energy() {
    int total = 0;
    for (int i = 0; i < m_SIZE; ++i) {
        total += m_spins[i] * (m_spins[m_neighbors[4*i + UP]] + m_spins[m_neighbors[4*i + DOWN]] +
                               m_spins[m_neighbors[4*i + LEFT]] + m_spins[m_neighbors[4*i + RIGHT]]);
    }
    return -total / 2.0;  // Convert to double when needed
}

double Ising2D::magnetization() const {
    double sum = 0.0;
    for (const auto s : m_spins) sum += s;
    return sum / m_SIZE;
}

void Ising2D::compute_metropolis_factors(double tstar) {
    m_h.resize(5);
    for (int de : {-8, -4, 0, 4, 8}) {
        m_h[(de+8)/4] = de <= 0 ? 1.0 : exp(-de/tstar);
    }
}
void Ising2D::metropolis_flip_spin(double tstar) {
    const int idx = m_ran_pos(m_gen);
    const int s = m_spins[idx];

    // Sum neighbors using precomputed indices
    const int* n = &m_neighbors[4*idx];
    const int sum = m_spins[n[0]] + m_spins[n[1]] + m_spins[n[2]] + m_spins[n[3]];

    const int deltaE = 2 * s * sum;
    const int h_idx = (deltaE + 8)/4;

    if (m_ran_u(m_gen) < m_h[h_idx]) {
        m_spins[idx] = -s;
        m_energy += deltaE;
    }
}

void Ising2D::save_current_config(int step_number) {
    if (m_config_save_path.empty()) return;
    
    std::string filename = m_config_save_path + "/step_" + std::to_string(step_number) + ".npy";
    auto config = get_configuration();
    
    cnpy::npy_save(
        filename,
        config.data(),
        {static_cast<size_t>(m_L), static_cast<size_t>(m_L)},
        "w"
    );
}

void Ising2D::set_config_save_path(const std::string& path) {
    m_config_save_path = path;
    // Create a snapshots subdirectory
    std::string snapshot_dir = path + "/snapshots";
    std::filesystem::create_directories(snapshot_dir);
    m_config_save_path = snapshot_dir;
}

void Ising2D::set_snapshot_interval(int interval) {
    m_snapshot_interval = interval;
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
    
    // Save configuration directly if path is set and interval is reached
    if (m_save_all_configs && !m_config_save_path.empty() && 
        (m_snapshot_count % m_snapshot_interval == 0)) {
        save_current_config(m_snapshot_count);
    }
    m_snapshot_count++;
}

void Ising2D::do_step_metropolis_mpi(double tstar, int N, MPI_Win win, int rank)
{
    compute_metropolis_factors(tstar);

    // Thermalization phase
    for (int i = 0; i < 1100; ++i) {
        metropolis_flip_spin(tstar);
    }
    // Measurement phase
    double mag_sum = 0, mag2_sum = 0, mag4_sum = 0;
    double ene_sum = 0, ene2_sum = 0, ene4_sum = 0;

    for (int i = 0; i < N; ++i) {
        // Perform 1100 Metropolis spin flips
        for (int j = 0; j < 1100; ++j) {
            metropolis_flip_spin(tstar);
        }

        // Measure magnetization, energy, etc.
        const double mag = fabs(magnetization());
        const double ene = m_energy;

        mag_sum += mag;
        mag2_sum += mag * mag;
        mag4_sum += mag * mag * mag * mag;
        ene_sum += ene;
        ene2_sum += ene * ene;
        ene4_sum += ene * ene * ene * ene;

        // Save configuration directly if needed
        if (m_save_all_configs && !m_config_save_path.empty() && (i % m_snapshot_interval == 0)) {
            save_current_config(i);
        }

        // Update progress after each 1000 measurements
        if ((i + 1) % 1000 == 0) {
            int one = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
            MPI_Win_unlock(0, win);
        }
    }

    // Finalize averages
    m_meanMag  = mag_sum / N;
    m_meanMag2 = mag2_sum / N;
    m_meanMag4 = mag4_sum / N;
    m_meanEne  = ene_sum / N;
    m_meanEne2 = ene2_sum / N;
    m_meanEne4 = ene4_sum / N;
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
}

void Ising2D::do_step_metropolis(double tstar, int N, int equ_N, int snapshot_interval) {
    compute_metropolis_factors(tstar);
    m_snapshot_count = 0;

    // Thermalization
    for (int i = 0; i < equ_N; ++i) {
        metropolis_flip_spin(tstar);
    }

    // Measurement phase
    double mag_sum = 0, mag2_sum = 0, mag4_sum = 0;
    double ene_sum = 0, ene2_sum = 0, ene4_sum = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 1100; ++j) {
            metropolis_flip_spin(tstar);
        }

        const double mag = fabs(magnetization());
        const double ene = m_energy;

        mag_sum += mag;
        mag2_sum += mag * mag;
        mag4_sum += mag * mag * mag * mag;
        ene_sum += ene;
        ene2_sum += ene * ene;
        ene4_sum += ene * ene * ene * ene;

        // Save configuration directly if enabled
        if (m_save_all_configs && !m_config_save_path.empty() && (i % snapshot_interval == 0)) {
            save_current_config(i);
        }
        m_snapshot_count = i;
    }

    // Store results
    m_meanMag = mag_sum / N;
    m_meanMag2 = mag2_sum / N;
    m_meanMag4 = mag4_sum / N;
    m_meanEne = ene_sum / N;
    m_meanEne2 = ene2_sum / N;
    m_meanEne4 = ene4_sum / N;
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
}

void Ising2D::thermalize_wolff(double tstar) {
    // Thermalize for ~5 full lattice sweeps
    for (int i = 0; i < 5 * m_SIZE; ++i) {
        wolff_cluster_update(1.0 - std::exp(-2.0 / tstar));
    }
}

void Ising2D::wolff_cluster_update(double p) {
    std::stack<int> stack;
    std::vector<bool> in_cluster(m_SIZE, false);
    int delta_energy = 0;  // Track energy change from flipped bonds

    const int start = m_ran_pos(m_gen);
    const char target_spin = m_spins[start];
    
    stack.push(start);
    in_cluster[start] = true;

    while (!stack.empty()) {
        const int current = stack.top();
        stack.pop();
        m_spins[current] = -target_spin;  // Flip the spin

        // Process neighbors
        const int* neighbors = &m_neighbors[4 * current];
        for (int i = 0; i < 4; ++i) {
            const int nidx = neighbors[i];
            if (!in_cluster[nidx]) {
                // Check if neighbor has the original spin (contributes to energy change)
                if (m_spins[nidx] == target_spin) {
                    delta_energy += 1;  // Each aligned bond adds +1 to energy
                }
                // Add to cluster with probability p if spin matches
                if (m_spins[nidx] == target_spin && m_ran_u(m_gen) < p) {
                    stack.push(nidx);
                    in_cluster[nidx] = true;
                }
            }
        }
    }

    m_energy += delta_energy;  // Correctly update energy
}

void Ising2D::do_step_wolff(double tstar, int N, int snapshot_interval) {
    const double p = 1.0 - std::exp(-2.0 / tstar);
    m_snapshot_count = 0;

    // Thermalize with periodic energy recalibration
    thermalize_wolff(tstar);
    m_energy = compute_energy();  // Ensure accurate starting energy

    // Measurement
    double mag_sum = 0, mag2_sum = 0, mag4_sum = 0;
    double ene_sum = 0, ene2_sum = 0, ene4_sum = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 12; ++j) {
            wolff_cluster_update(p);
        }

        // Recompute energy periodically to prevent drift
        if (i % 100 == 0) {
            m_energy = compute_energy();
        }

        const double mag = std::fabs(magnetization());
        const double ene = m_energy;

        mag_sum += mag;
        mag2_sum += mag * mag;
        mag4_sum += mag * mag * mag * mag;
        ene_sum += ene;
        ene2_sum += ene * ene;
        ene4_sum += ene * ene * ene * ene;

        // Save configuration directly if enabled
        if (m_save_all_configs && !m_config_save_path.empty() && (i % snapshot_interval == 0)) {
            save_current_config(i);
        }
        m_snapshot_count = i;
    }

    // Normalize results
    m_meanMag = mag_sum / N;
    m_meanMag2 = mag2_sum / N;
    m_meanMag4 = mag4_sum / N;
    m_meanEne = ene_sum / N;
    m_meanEne2 = ene2_sum / N;
    m_meanEne4 = ene4_sum / N;
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
}

void Ising2D::enable_save_all_configs(bool enable) {
    m_save_all_configs = enable;
}