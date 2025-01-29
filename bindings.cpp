#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ising.hpp"
#include "ising.cpp"
namespace py = pybind11;

PYBIND11_MODULE(pyising, m) {
    py::class_<Ising2D>(m, "Ising2D")
        .def(py::init<int, unsigned int>(), 
             py::arg("L"), 
             py::arg("seed") = 12345U)

        // Existing bindings...
        .def("initialize_spins", &Ising2D::initialize_spins)
        .def("compute_neighbors", &Ising2D::compute_neighbors)
        .def("compute_energy", &Ising2D::compute_energy)
        .def("magnetization", &Ising2D::magnetization)
        .def("do_step_metropolis", &Ising2D::do_step_metropolis)
        .def("do_step_wolff", &Ising2D::do_step_wolff)

        // NEW additions: single-step calls
        .def("do_metropolis_step", &Ising2D::do_metropolis_step)
        .def("do_wolff_step", &Ising2D::do_wolff_step)

        .def("get_configuration", &Ising2D::get_configuration)
        .def("get_L", &Ising2D::get_L)
        .def("get_magnetization", &Ising2D::get_magnetization)
        .def("get_magnetization2", &Ising2D::get_magnetization2)
        .def("get_magnetization4", &Ising2D::get_magnetization4)
        .def("get_energy_mean", &Ising2D::get_energy_mean)
        .def("get_energy2", &Ising2D::get_energy2)
        .def("get_energy4", &Ising2D::get_energy4)
        .def("get_binder_cumulant", &Ising2D::get_binder_cumulant);

    py::class_<Results>(m, "Results")
        .def_readonly("binder", &Results::binder)
        .def_readonly("meanMag", &Results::meanMag)
        .def_readonly("meanEne", &Results::meanEne);

    m.def("run_parallel_metropolis", &run_parallel_metropolis,
          py::arg("temps"), py::arg("L"), py::arg("N_steps"),
          py::arg("seed_base"), py::arg("use_wolff") = false);

}