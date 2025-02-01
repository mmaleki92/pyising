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
    int L, int N_steps,
    unsigned int seed_base,
    const std::string& output_dir,
    bool use_wolff,
    bool save_all_configs
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Distribute temperatures among ranks
    int num_temps = static_cast<int>(temps.size());
    int base_local_num = num_temps / size;
    int remainder = num_temps % size;
    int start = rank * base_local_num + std::min(rank, remainder);
    int local_num = base_local_num + (rank < remainder ? 1 : 0);
    int end = start + local_num;
    std::vector<double> local_temps(temps.begin() + start, temps.begin() + end);

    // Prepare local results
    std::vector<Results> local_results(local_temps.size());

    // Create output directory
    std::string L_dir = output_dir + "/L_" + std::to_string(L);
    std::filesystem::create_directories(L_dir);

    // Create an MPI window for global progress
    MPI_Win win;
    int* global_counter = nullptr;
    if (rank == 0) {
        MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &global_counter);
        *global_counter = 0;
        MPI_Win_create(global_counter, sizeof(int), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(nullptr, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    // Setup progress bar on rank 0
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
        bar.set_option(indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}
        });
        bar.set_option(indicators::option::ShowElapsedTime{true});
        bar.set_option(indicators::option::ShowRemainingTime{true});

        // We want the bar to go from 0 to num_temps (i.e., one increment per temperature).
        bar.set_option(indicators::option::MaxProgress{static_cast<size_t>(num_temps)});
    }

    #pragma omp parallel for
    for (size_t i = 0; i < local_temps.size(); ++i) {
        // Generate a seed based on rank, thread, and index
        unsigned int seed = seed_base + static_cast<unsigned int>(start + i);

        // Initialize Ising model
        Ising2D model(L, seed);
        model.initialize_spins();
        model.compute_neighbors();
        model.enable_save_all_configs(save_all_configs);

        // Perform simulation
        if (use_wolff) {
            model.do_step_wolff(local_temps[i], N_steps);
        } else {
            model.do_step_metropolis(local_temps[i], N_steps);
        }

        // Store results
        local_results[i] = model.get_results();
        local_results[i].T = local_temps[i];
        local_results[i].L = L;

        // Save config data
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << local_temps[i];
        std::string T_str = ss.str();
        std::string T_dir;
        {
            #pragma omp critical
            {
                T_dir = L_dir + "/T_" + T_str;
                std::filesystem::create_directories(T_dir);
            }
        }

        if (save_all_configs) {
            std::string all_filename = T_dir + "/all_configs.npy";
            const auto& all_configs = local_results[i].all_configurations;
            if (!all_configs.empty()) {
                size_t num_steps = all_configs.size();
                std::vector<int> flattened;
                flattened.reserve(num_steps * L * L);

                for (const auto& config : all_configs) {
                    flattened.insert(flattened.end(), config.begin(), config.end());
                }
                cnpy::npy_save(all_filename, flattened.data(), {num_steps, static_cast<size_t>(L), static_cast<size_t>(L)}, "w");
            }
        } else {
            std::string filename = T_dir + "/config.npy";
            const std::vector<int>& config = local_results[i].configuration;
            cnpy::npy_save(filename, config.data(), {static_cast<size_t>(L), static_cast<size_t>(L)}, "w");
        }

        // Done with this temperature -> increment global counter by 1
        #pragma omp critical
        {
            int one = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
            MPI_Win_unlock(0, win);
        }

        // Update progress bar if rank == 0
        if (rank == 0 && omp_get_thread_num() == 0) {
            static const int update_interval = 1; // update every temperature
            static int updates = 0;
            ++updates;
            if (updates % update_interval == 0) {
                MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
                int current_total = *global_counter;
                MPI_Win_unlock(0, win);
                bar.set_progress(current_total);
            }
        }
    }

    // Final update to ensure bar is complete
    if (rank == 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        bar.set_progress(*global_counter);
        MPI_Win_unlock(0, win);
        bar.mark_as_completed();
    }

    // MPI cleanup
    MPI_Win_free(&win);
    if (rank == 0) {
        MPI_Free_mem(global_counter);
    }

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
    res.configuration = get_configuration();
    res.all_configurations = m_all_configs;
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
      m_energy(0.0), m_save_all_configs(false) {}


void Ising2D::initialize_spins() {
    std::uniform_int_distribution<int> init_dist(0, 1);
    for (auto& s : m_spins) {
        s = init_dist(m_gen) ? 1 : -1;
    }
    compute_neighbors();
    m_energy = compute_energy();
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

void Ising2D::do_step_metropolis(double tstar, int N) {
    compute_metropolis_factors(tstar);

    // Thermalization (vectorized)
    for (int i = 0; i < 1100; ++i) {
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
        mag2_sum += mag*mag;
        mag4_sum += mag*mag*mag*mag;
        ene_sum += ene;
        ene2_sum += ene*ene;
        ene4_sum += ene*ene*ene*ene;
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
void Ising2D::do_step_wolff(double tstar, int N) {
    const double p = 1.0 - std::exp(-2.0 / tstar);

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

        if (m_save_all_configs) {
            m_all_configs.push_back(get_configuration());
        }
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
