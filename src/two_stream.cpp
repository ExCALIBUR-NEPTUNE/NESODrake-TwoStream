#include <neso_particles.hpp>
#include <pybind11/pybind11.h>
namespace py = pybind11;

inline int foo(){return 42;}

PYBIND11_MODULE(two_stream, m) {
  m.def("foo", &foo);
}
