#include <petsc.h>
#include <petscfe.h>
#include <petscdmtypes.h>
#include <neso_particles.hpp>
#include <pybind11/pybind11.h>
#include <cstdint> 
#include <cstring>

namespace py = pybind11;
using namespace NESO::Particles;

#ifndef CROSS_PRODUCT_3D
#define CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3) \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                        \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                        \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif


struct TwoStreamParticles {
  
  DM dm;
  int num_particles;
  double dt;
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<PetscInterface::DMPlexInterface> mesh;
  DomainSharedPtr domain;
  ParticleGroupSharedPtr particle_group;

  ParticleLoopSharedPtr loop_pbc;
  ParticleLoopSharedPtr loop_advect;
  std::shared_ptr<H5Part> h5part;

  TwoStreamParticles() = default;
  TwoStreamParticles(
    std::uintptr_t dm_vptr,
    const int num_particles,
    const double dt
  ) : 
    num_particles(num_particles),
    sycl_target(std::make_shared<SYCLTarget>(0, PETSC_COMM_WORLD)),
    dt(dt)
  {
    this->sycl_target->print_device_info();

    int flag;
    MPICHK(MPI_Initialized(&flag));
    NESOASSERT(flag, "MPI is not initalised.");
    
    DM dm_original = reinterpret_cast<DM>(dm_vptr);
    PETSCCHK(DMClone(dm_original, &this->dm));
    PetscBool is_initialised;
    PETSCCHK(PetscInitialized(&is_initialised));
    NESOASSERT(is_initialised, "PETSc is not initialised");

    this->mesh = 
      std::make_shared<PetscInterface::DMPlexInterface>(
        this->dm, 0, PETSC_COMM_WORLD);

    nprint("cell_count:", mesh->get_cell_count());

    auto mapper = std::make_shared<PetscInterface::DMPlexLocalMapper>(
        sycl_target, mesh);
    this->domain = std::make_shared<Domain>(mesh, mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), 2, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<REAL>("M"), 1),
                               ParticleProp(Sym<REAL>("V"), 3),
                               ParticleProp(Sym<REAL>("B"), 3),
                               ParticleProp(Sym<REAL>("E"), 3)};

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    this->loop_pbc = particle_loop(
      "TwoStreamParticles::pbc",
      this->particle_group,
      [=](auto P){
        P.at(0) = fmod(P.at(0) + 4.0, 1.0);
        P.at(1) = fmod(P.at(1) + 4.0, 1.0);
      },
      Access::write(Sym<REAL>("P"))
    );
    
    const REAL k_dt = this->dt;
    const REAL k_dht = 0.5 * k_dt; 
    this->loop_advect = particle_loop(
      "ParticleSystem:boris", this->particle_group,
      [=](auto B, auto E, auto Q, auto M, auto P, auto V) {
        const REAL QoM = Q.at(0) / M.at(0);

        const REAL scaling_t = QoM * k_dht;
        const REAL t_0 = B.at(0) * scaling_t;
        const REAL t_1 = B.at(1) * scaling_t;
        const REAL t_2 = B.at(2) * scaling_t;

        const REAL tmagsq = t_0 * t_0 + t_1 * t_1 + t_2 * t_2;
        const REAL scaling_s = 2.0 / (1.0 + tmagsq);

        const REAL s_0 = scaling_s * t_0;
        const REAL s_1 = scaling_s * t_1;
        const REAL s_2 = scaling_s * t_2;

        const REAL V_0 = V.at(0);
        const REAL V_1 = V.at(1);
        const REAL V_2 = V.at(2);

        // The E dat contains d(phi)/dx not E -> multiply by -1.
        const REAL v_minus_0 = V_0 + (-1.0 * E.at(0)) * scaling_t;
        const REAL v_minus_1 = V_1 + (-1.0 * E.at(1)) * scaling_t;
        const REAL v_minus_2 = V_2 + (-1.0 * E.at(2)) * scaling_t;

        REAL v_prime_0, v_prime_1, v_prime_2;
        CROSS_PRODUCT_3D(v_minus_0, v_minus_1, v_minus_2, t_0, t_1,
                         t_2, v_prime_0, v_prime_1, v_prime_2)

        v_prime_0 += v_minus_0;
        v_prime_1 += v_minus_1;
        v_prime_2 += v_minus_2;

        REAL v_plus_0, v_plus_1, v_plus_2;
        CROSS_PRODUCT_3D(v_prime_0, v_prime_1, v_prime_2, s_0, s_1,
                         s_2, v_plus_0, v_plus_1, v_plus_2)

        v_plus_0 += v_minus_0;
        v_plus_1 += v_minus_1;
        v_plus_2 += v_minus_2;

        // The E dat contains d(phi)/dx not E -> multiply by -1.
        V.at(0) = v_plus_0 + scaling_t * (-1.0 * E.at(0));
        V.at(1) = v_plus_1 + scaling_t * (-1.0 * E.at(1));
        V.at(2) = v_plus_2 + scaling_t * (-1.0 * E.at(2));

        // update of position to next time step
        P.at(0) += k_dt * V.at(0);
        P.at(1) += k_dt * V.at(1);
        // P.at(2) += k_dt * V.at(2);
      },
      Access::read(Sym<REAL>("B")), 
      Access::read(Sym<REAL>("E")),
      Access::read(Sym<REAL>("Q")), 
      Access::read(Sym<REAL>("M")),
      Access::write(Sym<REAL>("P")),
      Access::write(Sym<REAL>("V"))
    );

    this->add_particles();
  }

  void write(){
    if(!this->h5part){
      this->h5part = std::make_shared<H5Part>(
       "particle_trajectory.h5part", 
       this->particle_group,
       Sym<REAL>("P"),
       Sym<REAL>("V"), 
       Sym<REAL>("E"), 
       Sym<INT>("CELL_ID"),
       Sym<INT>("NESO_MPI_RANK"),
       Sym<INT>("PARTICLE_ID")
      );
    }
    this->h5part->write();
  }

  void free(){
    if (this->mesh){
      this->mesh->free();
    }
    PETSCCHK(DMDestroy(&this->dm));
    if (this->sycl_target){
      this->sycl_target->free();
    }
    if (this->h5part){
      this->h5part->close();
    }
  }

  void add_particles(){

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;

    get_decomp_1d(size, (long)this->num_particles, rank, &rstart, &rend);
    const long N = rend - rstart;
    
    std::srand(std::time(nullptr));
    int seed = std::rand();
    std::mt19937 rng_phasespace(seed + rank);
    
    double extents[2] = {1.0, 1.0};
    auto positions = uniform_within_extents(
      N, 2, extents, rng_phasespace);

    const REAL initial_velocity = 1.0;
    const REAL particle_charge = -1.0 / this->num_particles;
    const REAL particle_mass = 1.0 / this->num_particles;
  
    ParticleSet initial_distribution(
      N, this->particle_group->get_particle_spec());

    for (int px = 0; px < N; px++) {

      const bool species = px % 2;
      // x position
      const double pos_orig_0 = positions[0][px];
      initial_distribution[Sym<REAL>("P")][px][0] = pos_orig_0;

      // y position
      const double pos_orig_1 = positions[1][px];
      initial_distribution[Sym<REAL>("P")][px][1] = pos_orig_1;

      initial_distribution[Sym<REAL>("V")][px][0] =
          (species) ? initial_velocity : -1.0 * initial_velocity;
      initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
      initial_distribution[Sym<REAL>("Q")][px][0] = particle_charge;
      initial_distribution[Sym<REAL>("M")][px][0] = particle_mass;
    }

    this->particle_group->add_particles_local(initial_distribution);
    parallel_advection_initialisation(this->particle_group);
    parallel_advection_store(this->particle_group);
    const int num_steps = 20;
    for (int stepx = 0; stepx < num_steps; stepx++) {
      if(!rank){
        nprint("Advection Setup:", stepx);
      }
      // this->write();
      parallel_advection_step(this->particle_group, num_steps, stepx);
      this->transfer_particles();
    }
    parallel_advection_restore(this->particle_group);
    // Move particles to the owning ranks and correct cells.
    this->transfer_particles();
    // this->write();
  }
  
  void transfer_particles() {
    this->loop_pbc->execute();
    this->particle_group->hybrid_move();
    this->particle_group->cell_move();
  }

  void validate_halos(){
    NESOASSERT(this->mesh->validate_halos(), "Halo validation failed.");
  }

  void move(){
    this->loop_advect->execute();
    this->transfer_particles();
  }

};


PYBIND11_MODULE(two_stream, m) {

  py::class_<TwoStreamParticles>(m, "TwoStreamParticles")
      .def(py::init<std::uintptr_t, const int, const double>())
      .def("free", &TwoStreamParticles::free)
      .def("write",&TwoStreamParticles::write)
      .def("move",&TwoStreamParticles::move)
      .def("validate_halos",&TwoStreamParticles::validate_halos);

}
