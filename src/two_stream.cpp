#include <petsc.h>
#include <petscfe.h>
#include <petscdmtypes.h>
#include <neso_particles.hpp>
#include <pybind11/pybind11.h>
#include <cstdint> 
#include <cstring>
#include <cmath>

namespace py = pybind11;
using namespace NESO::Particles;

#ifndef CROSS_PRODUCT_3D
#define CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3) \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                        \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                        \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif

#ifndef KERNEL_ABS
#define KERNEL_ABS(x) ((x) < 0 ? (-(x)) : (x))
#endif

#ifndef KERNEL_MAX
#define KERNEL_MAX(x, y) ((x) < (y) ? ((y)) : (x))
#endif

struct TwoStreamParticles {
  
  DM dm;
  int num_particles;
  int ndim;
  double dt;
  double net_charge_density;
  int polynomial_order;
  std::string function_space;
  REAL particle_charge_dynamics;
  REAL particle_mass;

  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<PetscInterface::DMPlexInterface> mesh;
  DomainSharedPtr domain;
  ParticleGroupSharedPtr particle_group;

  ParticleLoopSharedPtr loop_pbc;
  std::shared_ptr<H5Part> h5part;

  std::shared_ptr<ExternalCommon::QuadraturePointMapper> qpm;
  std::shared_ptr<PetscInterface::DMPlexProjectEvaluate> dpe;

  TwoStreamParticles() = default;
  TwoStreamParticles(
    std::uintptr_t dm_vptr,
    const int num_particles,
    const double dt,
    const int polynomial_order
  ) : 
    num_particles(num_particles),
    sycl_target(std::make_shared<SYCLTarget>(0, PETSC_COMM_WORLD)),
    dt(dt),
    polynomial_order(polynomial_order)
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
    this->ndim = mesh->get_ndim();
    
    auto mapper = std::make_shared<PetscInterface::DMPlexLocalMapper>(
        sycl_target, mesh);
    this->domain = std::make_shared<Domain>(mesh, mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), 2, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<REAL>("PHI"), 1),
                               ParticleProp(Sym<REAL>("V"), 3),
                               ParticleProp(Sym<REAL>("B"), 3),
                               ParticleProp(Sym<REAL>("E0"), 1),
                               ParticleProp(Sym<REAL>("E1"), 1)
    };

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    auto mesh_bounding_box = mesh->dmh->get_bounding_box();
    REAL lower[3];
    REAL extent[3];

    for(int dx=0 ; dx<ndim ; dx++){
      lower[dx] = mesh_bounding_box->lower(dx);
      extent[dx] = mesh_bounding_box->upper(dx) - lower[dx];
      if (!this->sycl_target->comm_pair.rank_parent){
        nprint("lower:", lower[dx], "extent:", extent[dx]);
      }
    }
    
    NESOASSERT(ndim == 2, "unexpected ndim");
    const int k_ndim = ndim;
    this->loop_pbc = particle_loop(
      "TwoStreamParticles::pbc",
      this->particle_group,
      [=](auto P){
        const REAL offset = ((INT)(
            KERNEL_MAX(KERNEL_ABS(P.at(0)), KERNEL_ABS(P.at(1))) + 4.0
          )
        );
        for(int dx=0 ; dx<k_ndim ; dx++){
          P.at(dx) = Kernel::fmod(P.at(dx) + offset - lower[dx], extent[dx]) + lower[dx];
        }
      },
      Access::write(Sym<REAL>("P"))
    );
    
    this->qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);
    
    this->function_space = (this->polynomial_order == 0) ? "DG" : "Barycentric";

  }

  void write(){
    if(!this->h5part){
      this->h5part = std::make_shared<H5Part>(
       "particle_trajectory.h5part", 
       this->particle_group,
       Sym<REAL>("P"),
       Sym<REAL>("V"), 
       Sym<REAL>("E0"), 
       Sym<REAL>("E1"), 
       Sym<REAL>("Q"), 
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
    this->qpm->free();
  }

  REAL compute_potential_energy(){
    const REAL Q = this->particle_charge_dynamics;
    auto ga = std::make_shared<GlobalArray<REAL>>(this->sycl_target, 1);
    ga->fill(0.0);
    particle_loop(
      "compute_potential_energy",
      this->particle_group,
      [=](auto PHI, auto GA){
        GA.add(0, PHI.at(0) * Q);
      },
      Access::read(Sym<REAL>("PHI")),
      Access::add(ga)
    )->execute();
    return ga->get().at(0) * 0.5;
  }

  REAL compute_kinetic_energy(){
    const REAL M = this->particle_mass * 0.5;
    auto ga = std::make_shared<GlobalArray<REAL>>(this->sycl_target, 1);
    ga->fill(0.0);
    particle_loop(
      "compute_kinetic_energy",
      this->particle_group,
      [=](auto V, auto GA){
        GA.add(0,
          M * (V.at(0) * V.at(0) + V.at(1) * V.at(1))
        );
      },
      Access::read(Sym<REAL>("V")),
      Access::add(ga)
    )->execute();
    return ga->get().at(0);
  }

  void add_particles(){

    long rstart, rend;
    const long size = this->sycl_target->comm_pair.size_parent;
    const long rank = this->sycl_target->comm_pair.rank_parent;
    
    const int cell_count_local = mesh->get_cell_count();
    int cell_count_global;
    MPICHK(MPI_Allreduce(&cell_count_local, &cell_count_global, 1,
                  MPI_INT, MPI_SUM, this->sycl_target->comm_pair.comm_parent));
    
    const int npart_per_cell = std::ceil(((double) this->num_particles) / ((double) cell_count_global));
    this->num_particles = cell_count_global * npart_per_cell;
    const int num_particles_local = npart_per_cell * cell_count_local;
    const int num_particles_global = npart_per_cell * cell_count_global;
    const double volume = this->mesh->dmh->get_volume();

    if(!rank){
      nprint("npart_per_cell:", npart_per_cell);
      nprint("cell_count_local:", cell_count_local);
    }

    int global_id_start;
    MPICHK(MPI_Exscan(&num_particles_local, &global_id_start, 1,
           MPI_INT, MPI_SUM, this->sycl_target->comm_pair.comm_parent));
    
    std::srand(std::time(nullptr));
    int seed = std::rand();
    std::mt19937 rng_phasespace(seed + rank);
    
    std::vector<std::vector<double>> positions;
    std::vector<int> cells;
    PetscInterface::uniform_within_dmplex_cells(this->mesh, npart_per_cell, positions, cells);

    const REAL initial_velocity = 1.0;
    const REAL charge_density = 105.27578027828649;
    const REAL particle_number_density = 105.27578027828649;
    const REAL number_physical_particles = particle_number_density * volume;
    //const REAL particle_charge =
    //      charge_density * volume / number_physical_particles;

    const REAL particle_charge =
          charge_density * volume / num_particles_global;
    this->particle_charge_dynamics = charge_density * volume / number_physical_particles;

    this->particle_mass = 1.0;
  
    ParticleSet initial_distribution(
      num_particles_local, this->particle_group->get_particle_spec());

    std::normal_distribution norm_dist{0.0, 1.0};

    for (int px = 0; px < num_particles_local; px++) {
      const bool species = (global_id_start + px) % 2;
      // x position
      const double x0 = positions[0][px];
      initial_distribution[Sym<REAL>("P")][px][0] = x0;

      // y position
      const double x1 = positions[1][px];

      initial_distribution[Sym<REAL>("P")][px][1] = x1;
      initial_distribution[Sym<REAL>("Q")][px][0] = particle_charge;
      initial_distribution[Sym<REAL>("V")][px][0] =
          (species) ? initial_velocity : -1.0 * initial_velocity;
      initial_distribution[Sym<REAL>("V")][px][1] = 0.0;
      initial_distribution[Sym<REAL>("V")][px][2] = 0.0;
      initial_distribution[Sym<REAL>("B")][px][0] = 0.0;
      initial_distribution[Sym<REAL>("B")][px][1] = 0.0;
      initial_distribution[Sym<REAL>("B")][px][2] = 0.0;
      initial_distribution[Sym<INT>("PARTICLE_ID")][px][0] = global_id_start + px;
    }

    this->particle_group->add_particles_local(initial_distribution);
    this->transfer_particles();
    
    auto ga_net_charge = std::make_shared<GlobalArray<REAL>>(this->sycl_target, 1);
    ga_net_charge->fill(0.0);
    particle_loop(
      this->particle_group,
      [=](auto Q, auto NET){
        NET.add(0, Q.at(0));
      },
      Access::read(Sym<REAL>("Q")),
      Access::add(ga_net_charge)
    )->execute();
    
    this->net_charge_density = ga_net_charge->get().at(0) / volume;
    if(!rank){
      nprint("particle_mass:", particle_mass);
      nprint("particle_charge_dynamics:", this->particle_charge_dynamics);
      nprint("particle_charge_physical:", particle_charge);
      nprint("num_computational_particles_global:", num_particles_global);
      nprint("num_computational_particles_local:", num_particles_local);
      nprint("volume:", volume);
      nprint("total charge:", ga_net_charge->get().at(0));
      nprint("charge density:", this->net_charge_density);
    }

  }
  
  void transfer_particles() {
    this->loop_pbc->execute();
    this->particle_group->hybrid_move();
    this->particle_group->cell_move();
  }

  void validate_halos(){
    int rank;
    MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (mesh->dmh_halo){
      mesh->dmh_halo->write_vtk("halo_" + std::to_string(rank) + ".vtk");
    }
    NESOASSERT(this->mesh->validate_halos(), "Halo validation failed.");
  }

  double get_net_charge_density(){
    return this->net_charge_density;
  }

  void move_boris(){

    const REAL k_dt = this->dt;
    const REAL k_dht = 0.5 * k_dt; 
    const REAL QoM = this->particle_charge_dynamics / this->particle_mass;
    particle_loop(
      "ParticleSystem::move_boris", this->particle_group,
      [=](auto B, auto E0, auto E1, auto P, auto V) {

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
        const REAL v_minus_0 = V_0 + (-1.0 * E0.at(0)) * scaling_t;
        const REAL v_minus_1 = V_1 + (-1.0 * E1.at(0)) * scaling_t;
        const REAL v_minus_2 = V_2;

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
        V.at(0) = v_plus_0 + scaling_t * (-1.0 * E0.at(0));
        V.at(1) = v_plus_1 + scaling_t * (-1.0 * E1.at(0));
        V.at(2) = v_plus_2;

        // update of position to next time step
        P.at(0) += k_dt * V.at(0);
        P.at(1) += k_dt * V.at(1);
        // P.at(2) += k_dt * V.at(2);
      },
      Access::read(Sym<REAL>("B")), 
      Access::read(Sym<REAL>("E0")),
      Access::read(Sym<REAL>("E1")),
      Access::write(Sym<REAL>("P")),
      Access::write(Sym<REAL>("V"))
    )->execute();
    this->transfer_particles();
  }

  
  void move_vv1(){
    const REAL Q = this->particle_charge_dynamics;
    const REAL k_dt = this->dt;
    const REAL k_dht = 0.5 * k_dt; 
    const REAL k_M = this->particle_mass;
    const double dht_inverse_particle_mass = -1.0 * k_dht * Q / k_M;
    particle_loop(
      "ParticleSystem::move_vv1",
      this->particle_group,
      [=](
        auto P,
        auto V,
        auto E0,
        auto E1
      ){
          V.at(0) += E0.at(0) * dht_inverse_particle_mass;
          P.at(0) += k_dt * V.at(0);
          V.at(1) += E1.at(0) * dht_inverse_particle_mass;
          P.at(1) += k_dt * V.at(1);
      },
      Access::write(Sym<REAL>("P")),
      Access::write(Sym<REAL>("V")),
      Access::read(Sym<REAL>("E0")),
      Access::read(Sym<REAL>("E1"))
    )->execute();
    this->transfer_particles();
  }

  void move_vv2(){
    const REAL Q = this->particle_charge_dynamics;
    const REAL k_dt = this->dt;
    const REAL k_dht = 0.5 * k_dt; 
    const REAL k_M = this->particle_mass;
    const double dht_inverse_particle_mass = -1.0 * k_dht * Q / k_M;
    const int k_ndim = this->ndim;
    particle_loop(
      "ParticleSystem::move_vv2",
      this->particle_group,
      [=](
        auto V,
        auto E0,
        auto E1
      ){
          V.at(0) += E0.at(0) * dht_inverse_particle_mass;
          V.at(1) += E1.at(0) * dht_inverse_particle_mass;
      },
      Access::write(Sym<REAL>("V")),
      Access::read(Sym<REAL>("E0")),
      Access::read(Sym<REAL>("E1"))
    )->execute();
  }

  void setup_project(
    const int nrow,
    const int ncol,
    std::uintptr_t vptr
  ){
    NESOASSERT(ncol == this->ndim, "Quadrature points have a bad number of columns.");
    REAL * pts = reinterpret_cast<REAL *>(vptr);
    this->qpm->add_points_initialise();
    for (int rx = 0; rx<nrow; rx++) {
      this->qpm->add_point(pts);
      pts += ndim;
    }
    this->qpm->add_points_finalise();
    this->dpe = std::make_shared<PetscInterface::DMPlexProjectEvaluate>(
      qpm, function_space, polynomial_order);
  }

  void project(
    const std::string sym_name,
    const int nrow,
    const int ncol,
    std::uintptr_t vptr
    ){
    NESOASSERT(this->dpe != nullptr, "dpe is nullptr");
    this->dpe->project(this->particle_group, Sym<REAL>(sym_name));

    std::vector<REAL> output;
    this->qpm->get(1, output);
    REAL * pts = reinterpret_cast<REAL *>(vptr);
    std::memcpy(pts, output.data(), nrow * ncol * sizeof(REAL));
  }

  void evaluate(
    const std::string sym_name,
    const int nrow,
    const int ncol,
    std::uintptr_t vptr
  ){
    NESOASSERT(this->dpe != nullptr, "dpe_evaluate is nullptr");
    REAL * pts = reinterpret_cast<REAL *>(vptr);
    std::vector<REAL> input(nrow * ncol);
    for(int cx=0 ; cx <(nrow*ncol) ; cx++){
      input.at(cx) = pts[cx];
    }
    this->qpm->set(1, input);
    this->dpe->evaluate(
      this->particle_group, Sym<REAL>(sym_name));
  }
};

struct NESODrakeDebug {
  NESODrakeDebug(){
  }
  
  void foo(std::uintptr_t dm_vptr){
    nprint("foo start");
    DM dm = reinterpret_cast<DM>(dm_vptr);
    auto mesh = std::make_shared<PetscInterface::DMPlexInterface>(
      dm, 0, PETSC_COMM_WORLD);   

    mesh->free();
    nprint("foo end");
  }

};


PYBIND11_MODULE(two_stream, m) {

  py::class_<TwoStreamParticles>(m, "TwoStreamParticles")
      .def(py::init<std::uintptr_t, const int, const double, const int>())
      .def("free", &TwoStreamParticles::free)
      .def("write", &TwoStreamParticles::write)
      .def("move_boris", &TwoStreamParticles::move_boris)
      .def("move_vv1", &TwoStreamParticles::move_vv1)
      .def("move_vv2", &TwoStreamParticles::move_vv2)
      .def("validate_halos", &TwoStreamParticles::validate_halos)
      .def("add_particles", &TwoStreamParticles::add_particles)
      .def("setup_project", &TwoStreamParticles::setup_project)
      .def("project", &TwoStreamParticles::project)
      .def("evaluate", &TwoStreamParticles::evaluate)
      .def("compute_potential_energy", &TwoStreamParticles::compute_potential_energy)
      .def("compute_kinetic_energy", &TwoStreamParticles::compute_kinetic_energy)
      .def("get_net_charge_density", &TwoStreamParticles::get_net_charge_density);

  py::class_<NESODrakeDebug>(m, "NESODrakeDebug")
      .def(py::init<>())
      .def("foo", &NESODrakeDebug::foo);

}



