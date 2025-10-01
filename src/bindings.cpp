#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ising.hpp" // Include our pure C++ header

namespace py = pybind11;

// This wrapper function is the bridge between Python and C++.
py::list run_parallel_metropolis_py(
    const std::vector<double>& temps,
    int L,
    int N_steps,
    int equ_N,
    int snapshot_interval,
    unsigned int seed_base,
    const std::string& output_dir,
    bool use_wolff,
    bool save_all_configs)
{
    // Create a C++ vector to hold the results
    std::vector<Results> cpp_results;

    // Manually release the GIL only for the scope of this block.
    {
        py::gil_scoped_release release_gil;
        // 1. Call the pure C++ function while the GIL is released.
        cpp_results = run_simulation_cpp(
            temps, L, N_steps, equ_N, snapshot_interval, seed_base,
            output_dir, use_wolff, save_all_configs
        );
    } // <-- The GIL is automatically re-acquired here when 'release_gil' goes out of scope.

    // 2. Now, with the GIL held again, safely create Python objects.
    py::list final_results_list;
    for (const auto& res : cpp_results) {
        py::dict result_dict;
        result_dict["T"] = res.T;
        result_dict["L"] = res.L;
        result_dict["mean_mag"] = res.meanMag;
        result_dict["mean_ene"] = res.meanEne;
        result_dict["binder"] = res.binder;
        // NEW: Add new quantities to the Python dictionary
        result_dict["susceptibility"] = res.susceptibility;
        result_dict["specific_heat"] = res.specific_heat;
        result_dict["correlation_length"] = res.correlation_length;
        final_results_list.append(result_dict);
    }
    
    return final_results_list;
}


PYBIND11_MODULE(_pyising, m) {
    // Bind the Ising2D class as before
    py::class_<Ising2D>(m, "Ising2D")
        .def(py::init<int, unsigned int>(), 
             py::arg("L"), 
             py::arg("seed") = 12345U)
        .def("initialize_spins", &Ising2D::initialize_spins)
        .def("compute_neighbors", &Ising2D::compute_neighbors)
        .def("compute_energy", &Ising2D::compute_energy)
        .def("magnetization", &Ising2D::magnetization)
        .def("do_step_metropolis", &Ising2D::do_step_metropolis)
        .def("do_step_wolff", &Ising2D::do_step_wolff)
        .def("get_configuration", &Ising2D::get_configuration)
        .def("get_L", &Ising2D::get_L)
        .def("get_magnetization", &Ising2D::get_magnetization)
        .def("get_energy_mean", &Ising2D::get_energy_mean)
        .def("get_binder_cumulant", &Ising2D::get_binder_cumulant);

    // Bind the Python-visible name "run_parallel_metropolis" to our new wrapper.
    m.def("run_parallel_metropolis", &run_parallel_metropolis_py,
            "Run the parallel Metropolis simulation",
            py::arg("temps"), py::arg("L"), py::arg("N_steps"),
            py::arg("equ_N"), py::arg("snapshot_interval"),
            py::arg("seed_base"), py::arg("output_dir"),
            py::arg("use_wolff"), py::arg("save_all_configs")
        );
}