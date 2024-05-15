#include <petsc.h>
#include <petscfe.h>
#include <petscdmtypes.h>
#include <neso_particles.hpp>
#include <pybind11/pybind11.h>
#include <cstdint> 
#include <cstring>

namespace py = pybind11;
using namespace NESO::Particles;

struct TwoStreamParticles {
  
  int num_particles;
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<PetscInterface::DMPlexInterface> mesh_interface;
  DM dm;

  TwoStreamParticles() = default;
  TwoStreamParticles(
    std::uintptr_t dm_vptr,
    const int num_particles
  ) : 
    num_particles(num_particles),
    sycl_target(std::make_shared<SYCLTarget>(0, PETSC_COMM_WORLD))
  {
    this->sycl_target->print_device_info();

    int flag;
    MPICHK(MPI_Initialized(&flag));
    NESOASSERT(flag, "MPI is not initalised.");

    this->dm = reinterpret_cast<DM>(dm_vptr);
    PetscBool is_initialised;
    PETSCCHK(PetscInitialized(&is_initialised));
    NESOASSERT(is_initialised, "PETSc is not initialised");

  }

  void free(){
    if (this->sycl_target){
      this->sycl_target->free();
    }
  }
};


PYBIND11_MODULE(two_stream, m) {

  py::class_<TwoStreamParticles>(m, "TwoStreamParticles")
      .def(py::init<std::uintptr_t, const int>())
      .def("free", &TwoStreamParticles::free);


}
