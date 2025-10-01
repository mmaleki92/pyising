#include <mpi.h>
#include <omp.h>
#include <stack>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include "ising.hpp"
#include <complex>
#include "fft.hpp"
#include "cnpy/cnpy.h"
#include <indicators/progress_bar.hpp>
#include <indicators/termcolor.hpp>

#define UP    0
#define RIGHT 1
#define LEFT  2
#define DOWN  3

std::vector<Results> run_simulation_cpp(
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_temps = static_cast<int>(temps.size());
    int base_local_num = num_temps / size;
    int remainder = num_temps % size;
    int start = rank * base_local_num + std::min(rank, remainder);
    int local_num = base_local_num + (rank < remainder ? 1 : 0);
    int end = start + local_num;
    std::vector<double> local_temps(temps.begin() + start, temps.begin() + end);

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
        bar.set_option(indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
        bar.set_option(indicators::option::ShowElapsedTime{true});
        bar.set_option(indicators::option::ShowRemainingTime{true});
        bar.set_option(indicators::option::MaxProgress{static_cast<size_t>(num_temps)});
    }

    std::string L_dir = output_dir + "/L_" + std::to_string(L);
    if (rank == 0) std::filesystem::create_directories(L_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win win;
    int* global_counter = nullptr;
    if (rank == 0) {
        MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &global_counter);
        *global_counter = 0;
        MPI_Win_create(global_counter, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(nullptr, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    std::vector<Results> local_results(local_temps.size());

    #pragma omp parallel for
    for (size_t i = 0; i < local_temps.size(); ++i) {
        unsigned int seed = seed_base + static_cast<unsigned int>(start + i);
        Ising2D model(L, seed);
        model.initialize_spins();
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << local_temps[i];
        std::string T_str = ss.str();
        std::string T_dir = L_dir + "/T_" + T_str;
        
        if (save_all_configs) {
            std::filesystem::create_directories(T_dir);
            model.set_config_save_path(T_dir);
            model.enable_save_all_configs(true);
        }
        
        if (use_wolff) {
            model.do_step_wolff(local_temps[i], N_steps, snapshot_interval);
        } else {
            model.do_step_metropolis(local_temps[i], N_steps, equ_N, snapshot_interval);
        }

        local_results[i] = model.get_results();
        local_results[i].T = local_temps[i];

        #pragma omp critical (progress_update)
        {
            int one = 1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Accumulate(&one, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
            MPI_Win_unlock(0, win);

            if (rank == 0) {
                MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
                int current_total = *global_counter;
                MPI_Win_unlock(0, win);
                bar.set_progress(current_total);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        bar.set_progress(num_temps);
        bar.mark_as_completed();
    }
    
    MPI_Win_free(&win);
    if (rank == 0 && global_counter != nullptr) MPI_Free_mem(global_counter);

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
    res.L = m_L;
    // NEW: Populate new results
    res.susceptibility = m_susceptibility;
    res.specific_heat = m_specificHeat;
    res.correlation_length = m_correlationLength;
    res.correlation_function = m_correlation_function;
    return res;
}

std::vector<int> Ising2D::get_configuration() const {
    std::vector<int> config(m_spins.begin(), m_spins.end());
    return config;
}

Ising2D::Ising2D(int L, unsigned int seed)
    : m_L(L), m_SIZE(L*L),
      m_gen(seed), m_ran_pos(0, L*L-1), m_ran_u(0.0, 1.0),
      m_spins(L*L, 1), m_neighbors(4*L*L, 0),
      m_energy(0.0), m_save_all_configs(false),
      m_snapshot_count(0), m_snapshot_interval(100),
      m_config_save_path("") {
    precompute_trig_factors(); // NEW: Precompute factors for correlation length
}

// NEW: Precompute trigonometric factors for calculating the structure factor S(k_min)
void Ising2D::precompute_trig_factors() {
    m_cos_kx.resize(m_SIZE);
    m_sin_kx.resize(m_SIZE);
    m_cos_ky.resize(m_SIZE);
    m_sin_ky.resize(m_SIZE);
    double k_min = 2.0 * M_PI / m_L;
    for (int y = 0; y < m_L; ++y) {
        for (int x = 0; x < m_L; ++x) {
            int idx = y * m_L + x;
            m_cos_kx[idx] = cos(k_min * x);
            m_sin_kx[idx] = sin(k_min * x);
            m_cos_ky[idx] = cos(k_min * y);
            m_sin_ky[idx] = sin(k_min * y);
        }
    }
}

std::vector<double> Ising2D::calculate_correlation_function() const {
    // 1. Copy spin configuration to a complex vector for FFT
    std::vector<std::complex<double>> fft_grid(m_SIZE);
    for(int i = 0; i < m_SIZE; ++i) {
        fft_grid[i] = std::complex<double>(m_spins[i], 0.0);
    }

    // 2. Perform 2D Fast Fourier Transform
    fft2d(fft_grid, m_L, false);

    // 3. Compute the power spectrum |FFT(s)|^2
    for(int i = 0; i < m_SIZE; ++i) {
        fft_grid[i] = fft_grid[i] * std::conj(fft_grid[i]);
    }

    // 4. Perform inverse 2D FFT to get the 2D correlation map
    fft2d(fft_grid, m_L, true);

    // 5. Radially average the 2D correlation map to get 1D G(r)
    int max_r = m_L / 2;
    std::vector<double> G_r(max_r + 1, 0.0);
    std::vector<long> counts(max_r + 1, 0);

    for (int y = 0; y < m_L; ++y) {
        for (int x = 0; x < m_L; ++x) {
            // Use periodic boundary conditions for distance
            int dx = (x > m_L/2) ? m_L - x : x;
            int dy = (y > m_L/2) ? m_L - y : y;
            int r = static_cast<int>(std::round(std::sqrt(dx*dx + dy*dy)));

            if (r <= max_r) {
                // The real part of the IFFT result is what we need
                G_r[r] += fft_grid[y * m_L + x].real();
                counts[r]++;
            }
        }
    }
    
    // Normalize by the number of pairs for each distance r and by N
    for (int r = 0; r <= max_r; ++r) {
        if (counts[r] > 0) {
            G_r[r] /= (counts[r] * m_SIZE);
        }
    }
    return G_r;
}

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
            m_neighbors[4*idx]     = y * m_L + wrap(x + 1);
            m_neighbors[4*idx + 1] = y * m_L + wrap(x - 1);
            m_neighbors[4*idx + 2] = wrap(y + 1) * m_L + x;
            m_neighbors[4*idx + 3] = wrap(y - 1) * m_L + x;
        }
    }
}

double Ising2D::compute_energy() {
    int total = 0;
    for (int i = 0; i < m_SIZE; ++i) {
        total += m_spins[i] * (m_spins[m_neighbors[4*i + UP]] + m_spins[m_neighbors[4*i + DOWN]] +
                               m_spins[m_neighbors[4*i + LEFT]] + m_spins[m_neighbors[4*i + RIGHT]]);
    }
    return -total / 2.0;
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
    cnpy::npy_save(filename, config.data(), {static_cast<size_t>(m_L), static_cast<size_t>(m_L)}, "w");
}

void Ising2D::set_config_save_path(const std::string& path) {
    m_config_save_path = path;
}

void Ising2D::set_snapshot_interval(int interval) {
    m_snapshot_interval = interval;
}

void Ising2D::do_step_metropolis(double tstar, int N, int equ_N, int snapshot_interval) {
    compute_metropolis_factors(tstar);
    m_snapshot_count = 0;
    for (int i = 0; i < equ_N; ++i) {
        metropolis_flip_spin(tstar);
    }

    double mag_sum = 0, mag2_sum = 0, mag4_sum = 0;
    double ene_sum = 0, ene2_sum = 0, ene4_sum = 0;
    double s_k_min_sum = 0;
    // Accumulator for the correlation function
    std::vector<double> G_r_sum(m_L / 2 + 1, 0.0);

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

        // NEW: Calculate structure factor for this configuration
        double cos_sum_x = 0.0, sin_sum_x = 0.0;
        double cos_sum_y = 0.0, sin_sum_y = 0.0;
        for (int site = 0; site < m_SIZE; ++site) {
            cos_sum_x += m_spins[site] * m_cos_kx[site];
            sin_sum_x += m_spins[site] * m_sin_kx[site];
            cos_sum_y += m_spins[site] * m_cos_ky[site];
            sin_sum_y += m_spins[site] * m_sin_ky[site];
        }
        double s_k_x = (cos_sum_x * cos_sum_x + sin_sum_x * sin_sum_x) / m_SIZE;
        double s_k_y = (cos_sum_y * cos_sum_y + sin_sum_y * sin_sum_y) / m_SIZE;
        s_k_min_sum += (s_k_x + s_k_y) / 2.0; // Average over both kx and ky directions

        // NEW: Calculate G(r) for this step and add to sum
        std::vector<double> G_r_current = calculate_correlation_function();
        for(size_t r = 0; r < G_r_sum.size(); ++r) {
            G_r_sum[r] += G_r_current[r];
        }

        if (m_save_all_configs && !m_config_save_path.empty() && (i % snapshot_interval == 0)) {
            save_current_config(i);
        }
        m_snapshot_count = i;    }
    
    double N_double = static_cast<double>(N);
    m_meanMag = mag_sum / N_double;
    m_meanMag2 = mag2_sum / N_double;
    m_meanMag4 = mag4_sum / N_double;
    m_meanEne = ene_sum / N_double;
    m_meanEne2 = ene2_sum / N_double;
    m_meanEne4 = ene4_sum / N_double;
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));
    // Magnetic Susceptibility
    m_susceptibility = (m_SIZE / tstar) * (m_meanMag2 - m_meanMag * m_meanMag);
    // Specific Heat
    m_specificHeat = (1.0 / (m_SIZE * tstar * tstar)) * (m_meanEne2 - m_meanEne * m_meanEne);
    // Correlation Length
    double mean_s_k_min = s_k_min_sum / N_double;
    double term_in_sqrt = ( (m_SIZE * m_meanMag2) / mean_s_k_min ) - 1.0;
    if (term_in_sqrt > 0 && mean_s_k_min > 1e-12) {
        m_correlationLength = (1.0 / (2.0 * sin(M_PI / m_L))) * sqrt(term_in_sqrt);
    } else {
        m_correlationLength = 0.0;
    }
    // Finalize the connected correlation function
    m_correlation_function.resize(G_r_sum.size());
    double avg_mag_sq = m_meanMag * m_meanMag;
    for(size_t r = 0; r < G_r_sum.size(); ++r) {
        // Average G(r) and subtract <S>^2
        m_correlation_function[r] = (G_r_sum[r] / N_double) - avg_mag_sq;
    }
}

void Ising2D::thermalize_wolff(double tstar) {
    for (int i = 0; i < 5 * m_SIZE; ++i) {
        wolff_cluster_update(1.0 - std::exp(-2.0 / tstar));
    }
}

void Ising2D::wolff_cluster_update(double p) {
    std::stack<int> stack;
    std::vector<bool> in_cluster(m_SIZE, false);
    const int start = m_ran_pos(m_gen);
    const char target_spin = m_spins[start];
    stack.push(start);
    in_cluster[start] = true;

    while (!stack.empty()) {
        const int current = stack.top();
        stack.pop();
        m_spins[current] = -target_spin;
        const int* neighbors = &m_neighbors[4 * current];
        for (int i = 0; i < 4; ++i) {
            const int nidx = neighbors[i];
            if (!in_cluster[nidx] && m_spins[nidx] == target_spin && m_ran_u(m_gen) < p) {
                stack.push(nidx);
                in_cluster[nidx] = true;
            }
        }
    }
    m_energy = compute_energy();
}

void Ising2D::do_step_wolff(double tstar, int N, int snapshot_interval) {
    const double p = 1.0 - std::exp(-2.0 / tstar);
    m_snapshot_count = 0;
    thermalize_wolff(tstar);
    m_energy = compute_energy();

    double mag_sum = 0, mag2_sum = 0, mag4_sum = 0;
    double ene_sum = 0, ene2_sum = 0, ene4_sum = 0;
    double s_k_min_sum = 0;
    // Accumulator for the correlation function
    std::vector<double> G_r_sum(m_L / 2 + 1, 0.0);

    for (int i = 0; i < N; ++i) {
        wolff_cluster_update(p);
        const double mag = std::fabs(magnetization());
        const double ene = m_energy;
        mag_sum += mag;
        mag2_sum += mag * mag;
        mag4_sum += mag * mag * mag * mag;
        ene_sum += ene;
        ene2_sum += ene * ene;
        ene4_sum += ene * ene * ene * ene;


        // Calculate structure factor for this configuration
        double cos_sum_x = 0.0, sin_sum_x = 0.0;
        double cos_sum_y = 0.0, sin_sum_y = 0.0;
        for (int site = 0; site < m_SIZE; ++site) {
            cos_sum_x += m_spins[site] * m_cos_kx[site];
            sin_sum_x += m_spins[site] * m_sin_kx[site];
            cos_sum_y += m_spins[site] * m_cos_ky[site];
            sin_sum_y += m_spins[site] * m_sin_ky[site];
        }
        double s_k_x = (cos_sum_x * cos_sum_x + sin_sum_x * sin_sum_x) / m_SIZE;
        double s_k_y = (cos_sum_y * cos_sum_y + sin_sum_y * sin_sum_y) / m_SIZE;
        s_k_min_sum += (s_k_x + s_k_y) / 2.0; // Average over both kx and ky directions

        // Calculate G(r) for this step and add to sum
        std::vector<double> G_r_current = calculate_correlation_function();
        for(size_t r = 0; r < G_r_sum.size(); ++r) {
            G_r_sum[r] += G_r_current[r];
        }

        if (m_save_all_configs && !m_config_save_path.empty() && (i % snapshot_interval == 0)) {
            save_current_config(i);
        }
        m_snapshot_count = i;
    }
    
    double N_double = static_cast<double>(N);
    m_meanMag = mag_sum / N_double;
    m_meanMag2 = mag2_sum / N_double;
    m_meanMag4 = mag4_sum / N_double;
    m_meanEne = ene_sum / N_double;
    m_meanEne2 = ene2_sum / N_double;
    m_meanEne4 = ene4_sum / N_double;
    m_binder = 1.0 - (m_meanMag4 / (3.0 * m_meanMag2 * m_meanMag2));

    // NEW: Finalize the connected correlation function
    m_correlation_function.resize(G_r_sum.size());

    // Magnetic Susceptibility
    m_susceptibility = (m_SIZE / tstar) * (m_meanMag2 - m_meanMag * m_meanMag);
    // Specific Heat
    m_specificHeat = (1.0 / (m_SIZE * tstar * tstar)) * (m_meanEne2 - m_meanEne * m_meanEne);
    // Correlation Length
    double mean_s_k_min = s_k_min_sum / N_double;
    double term_in_sqrt = ( (m_SIZE * m_meanMag2) / mean_s_k_min ) - 1.0;
    if (term_in_sqrt > 0 && mean_s_k_min > 1e-12) {
        m_correlationLength = (1.0 / (2.0 * sin(M_PI / m_L))) * sqrt(term_in_sqrt);
    } else {
        m_correlationLength = 0.0;
    }
    double avg_mag_sq = m_meanMag * m_meanMag;
    for(size_t r = 0; r < G_r_sum.size(); ++r) {
        // Average G(r) and subtract <S>^2
        m_correlation_function[r] = (G_r_sum[r] / N_double) - avg_mag_sq;
    }
}
void Ising2D::enable_save_all_configs(bool enable) {
    m_save_all_configs = enable;
}
