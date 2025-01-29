#include "ising.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

#define UP    0
#define RIGHT 1
#define LEFT  2
#define DOWN  3

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
    for (int i = 0; i < m_SIZE; i++)
    {
        m_spins[i] = (m_brandom(m_gen) == 1); 
    }
}

void Ising2D::compute_neighbors()
{
    for (int i = 0; i < m_L; i++)
    {
        for (int j = 0; j < m_L; j++)
        {
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
    for (int i = 0; i < m_SIZE; i++)
    {
        int sum_neigh = spin_val(m_spins[m_neighbors[i][UP]]) 
                      + spin_val(m_spins[m_neighbors[i][DOWN]]) 
                      + spin_val(m_spins[m_neighbors[i][RIGHT]]) 
                      + spin_val(m_spins[m_neighbors[i][LEFT]]);
        // local energy = -spin[i]*(sum_of_neighbors)
        // but the code as provided uses a different counting for "delta"
        // We'll keep consistent with original approach: energy for spin i
        // was effectively 2 * sum_neigh - 4 for (spin=+1) or 4 - 2 * sum_neigh for (spin=-1).
        // That equals - spin[i]* ( sum_of_neighbors*(2) ) + constant.
        // We'll just replicate the old formula:

        if (m_spins[i]) // spin = +1
            totalEnergy += (2 * sum_neigh - 4);
        else            // spin = -1
            totalEnergy += (4 - 2 * sum_neigh);
    }
    // The original code scaled by 2.0 / SIZE
    return 2.0 * totalEnergy / (1.0 * m_SIZE);
}

double Ising2D::magnetization() const
{
    double sum = 0.0;
    for (int i = 0; i < m_SIZE; i++)
    {
        sum += spin_val(m_spins[i]);
    }
    return (sum / (double)m_SIZE);
}

void Ising2D::compute_metropolis_factors(double tstar)
{
    // h[i] for i in [-4,-2,0,2,4] => index = (i+4)/2 => h[ (deltaE +4)/2 ]
    // deltaE from flipping one spin can be in [-4, -2, 0, 2, 4].
    // The original code used: exp(-2*i/tstar) to fill h
    for (int i = -4; i <= 4; i += 2)
    {
        double val = std::exp((-2.0 * i)/tstar);
        m_h[(i + 4) / 2] = (val < 1.0) ? val : 1.0;
    }
}

void Ising2D::metropolis_flip_spin()
{
    int index = m_ran_pos(m_gen);
    // sum of neighbors in +1/-1 form
    int sum_neigh = spin_val(m_spins[m_neighbors[index][UP]]) 
                  + spin_val(m_spins[m_neighbors[index][DOWN]]) 
                  + spin_val(m_spins[m_neighbors[index][RIGHT]]) 
                  + spin_val(m_spins[m_neighbors[index][LEFT]]);

    // spin before flip in +1/-1
    int currentSpin = spin_val(m_spins[index]);
    // if spin = +1 => dE = deltaE = 2*(sum_neigh) - 4
    // if spin = -1 => dE = 4 - 2*(sum_neigh)
    int deltaE = 0;
    if (currentSpin == +1)
        deltaE = (2 * sum_neigh - 4);
    else
        deltaE = (4 - 2 * sum_neigh);

    int idx = (deltaE + 4)/2; // to index m_h
    if (m_ran_u(m_gen) < m_h[idx])
    {
        // Accept flip
        m_spins[index] = !m_spins[index];
        // Update total energy
        m_energy += (2.0 * deltaE) / (1.0 * m_SIZE);
    }
}

void Ising2D::thermalize_metropolis(double tstar)
{
    // Example: 1100 steps was used in original code
    // This is arbitrary, you can adjust as needed
    for (int i = 0; i < 1100; i++)
    {
        metropolis_flip_spin();
    }
}

void Ising2D::measure_observables(double N)
{
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
}

void Ising2D::do_step_metropolis(double tstar, int N)
{
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
    for (int i = 0; i < N; i++)
    {
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

void Ising2D::thermalize_wolff(double tstar)
{
    // For example ~15 cluster updates, from original code
    for (int i = 0; i < 15; i++)
    {
        int pos = m_ran_pos(m_gen);
        double p = 1.0 - std::exp(-2.0 / tstar);
        wolff_add_to_cluster(pos, p);
    }
}

void Ising2D::wolff_add_to_cluster(int pos, double p)
{
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
    for (int i = 0; i < 4; i++)
    {
        int npos = m_neighbors[pos][i];
        if (spin_val(m_spins[npos]) == oldSpinVal)
        {
            // Probability p
            if (m_ran_u(m_gen) < p)
            {
                wolff_add_to_cluster(npos, p);
            }
        }
    }
}

void Ising2D::do_step_wolff(double tstar, int N)
{
    m_meanMag  = 0.0;
    m_meanMag2 = 0.0;
    m_meanMag4 = 0.0;
    m_meanEne  = 0.0;
    m_meanEne2 = 0.0;
    m_meanEne4 = 0.0;
    m_binder   = 0.0;

    // Thermalize with some cluster updates
    thermalize_wolff(tstar);

    for (int i = 0; i < N; i++)
    {
        // Each iteration do ~12 cluster updates
        for (int j = 0; j < 12; j++)
        {
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