#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ising.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyising, m) {
    py::class_<Ising2D>(m, "Ising2D")
        .def(py::init<int, unsigned int>(), 
             py::arg("L"), 
             py::arg("seed") = 12345U)
        .def("initialize_spins", &Ising2D::initialize_spins)
        .def("compute_neighbors", &Ising2D::compute_neighbors)
        .def("compute_energy", &Ising2D::compute_energy)
        .def("magnetization", &Ising2D::magnetization)
        .def("do_step_metropolis", &Ising2D::do_step_metropolis,
             py::arg("tstar"), py::arg("N"))
        .def("do_step_wolff", &Ising2D::do_step_wolff,
             py::arg("tstar"), py::arg("N"))
        .def("get_L", &Ising2D::get_L)
        .def("get_magnetization", &Ising2D::get_magnetization)
        .def("get_magnetization2", &Ising2D::get_magnetization2)
        .def("get_magnetization4", &Ising2D::get_magnetization4)
        .def("get_energy_mean", &Ising2D::get_energy_mean)
        .def("get_energy2", &Ising2D::get_energy2)
        .def("get_energy4", &Ising2D::get_energy4)
        .def("get_binder_cumulant", &Ising2D::get_binder_cumulant);
}