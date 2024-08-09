from firedrake import *
from two_stream import *
from firedrake.__future__ import interpolate, Interpolator
import sys
import time


class ProjectEvaluate:
    def __init__(self, particle_state, V):
        self.particle_state = particle_state
        self.V = V
        m = V.mesh()
        W = VectorFunctionSpace(m, V.ufl_element())
        X = assemble(interpolate(m.coordinates, W))
        quadrature_points = X.dat.data_ro
        nrow, ncol = quadrature_points.shape
        ptr = X.dat.data_ro.ctypes._as_parameter_.value
        self.particle_state.setup_project(nrow, ncol, ptr)

    def evaluate(self, u, sym_name):
        space = u.dat.data_ro
        nrow = space.shape[0]
        ncol = 1
        self.particle_state.evaluate(sym_name, nrow, ncol, space.ctypes._as_parameter_.value)

    def project(self, sym_name, u):
        space = u.dat.data
        nrow = space.shape[0]
        ncol = 1
        self.particle_state.project(sym_name, nrow, ncol, space.ctypes._as_parameter_.value)


class Projector:
    def __init__(self, project_evaluate, sym_name, u):
        self.project_evaluate = project_evaluate
        self.sym_name = sym_name
        self.u = u
        self.v = Function(self.project_evaluate.V)
        self.interpolator = Interpolator(self.v, self.u.function_space())

    def __call__(self):
        self.project_evaluate.project(self.sym_name, self.v)
        self.u.assign(assemble(self.interpolator.interpolate()))


class Evaluator:
    def __init__(self, project_evaluate, u, sym_name):
        self.sym_name = sym_name
        self.project_evaluate = project_evaluate
        self.interpolator = Interpolator(u, self.project_evaluate.V)
        self.v = Function(self.project_evaluate.V)

    def __call__(self):
        self.v.assign(assemble(self.interpolator.interpolate()))
        self.project_evaluate.evaluate(self.v, self.sym_name)

boris = "boris"
velocity_verlet = "velocity_verlet"

if __name__ == "__main__":
    
    integrator = boris
    num_steps = 4000
    num_print_steps = 10
    num_write_steps = 20
    num_energy_steps = 10

    num_cells_y = 3
    num_cells_x = 100
    num_particles = 400000
    dt = 0.001
    p = 1
    mesh_width = 0.01


    mesh_np = RectangleMesh(num_cells_x, num_cells_y, 1.0, mesh_width)
    DG_np = FunctionSpace(mesh_np, "DG", p)
    particle_state = TwoStreamParticles(mesh_np.topology_dm.handle, num_particles, dt, p)
    pe = ProjectEvaluate(particle_state, DG_np)

    mesh = PeriodicRectangleMesh(num_cells_x, num_cells_y, 1.0, mesh_width)
    BDM = FunctionSpace(mesh, "BDM", p + 1)
    DG = FunctionSpace(mesh, "DG", p)
    W = BDM * DG

    particle_state.validate_halos()
    particle_state.add_particles()

    E = Function(BDM)
    rho = Function(DG, name="rho")
    neutralising_field = Function(DG, name="neutralising_field")
    poisson_rhs = Function(DG, name="poisson_rhs")
    net_charge_density = particle_state.get_net_charge_density()
    neutralising_field.interpolate(net_charge_density)

    poisson_rhs.interpolate(neutralising_field - rho)

    # weak form
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
    L = poisson_rhs * v * dx

    w = Function(W)
    E, phi = w.subfunctions
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    projector_rho = Projector(pe, "Q", rho)
    evaluator_E0 = Evaluator(pe, E[0], "E0")
    evaluator_E1 = Evaluator(pe, E[1], "E1")
    evaluator_phi = Evaluator(pe, phi, "PHI")

    projector_rho()
    poisson_rhs.interpolate(neutralising_field - rho)

    rho_integral = assemble(rho * dx)
    rhs_integral = assemble(poisson_rhs * dx)

    if mpi.COMM_WORLD.rank == 0:
        print("integrator:", integrator)
        print("rho_integral:", rho_integral)
        print("rhs_integral:", rhs_integral)

    out_rho = VTKFile("rho.pvd")
    out_E = VTKFile("E.pvd")

    list_t = []
    list_E2 = []
    potential_energy = []
    kinetic_energy = []
    total_energy = []

    def update_particle_energy():
        pe = particle_state.compute_potential_energy()
        ke = particle_state.compute_kinetic_energy()
        potential_energy.append(pe)
        kinetic_energy.append(ke)
        total_energy.append(pe + ke)

    assert integrator == velocity_verlet or integrator == boris, "Unknown integrator."
    t0 = time.time()
    for stepx in range(num_steps):
        if integrator == velocity_verlet:
            particle_state.move_vv1()
        projector_rho()
        poisson_rhs.interpolate(neutralising_field - rho)
        solve(a == L, w, bcs=[], nullspace=nullspace)
        evaluator_E0()
        evaluator_E1()
        if integrator == velocity_verlet:
            particle_state.move_vv2()
        elif integrator == boris:
            particle_state.move_boris()

        if (stepx % num_write_steps == 0) and (num_write_steps > 0):
            out_rho.write(rho, poisson_rhs)
            out_E.write(E)
            particle_state.write()

        last_potential_energy = 0.0
        last_kinetic_energy = 0.0

        if stepx % num_energy_steps == 0:
            list_t.append(stepx * dt)
            list_E2.append(norm(E) ** 2.0)
            evaluator_phi()
            update_particle_energy()

        if stepx % num_print_steps == 0:
            if mpi.COMM_WORLD.rank == 0:
                time_per_step = (time.time() - t0) / (stepx + 1)
                print(
                    "step: {:5d}".format(stepx),
                    "time per step: {:5.2e}".format(time_per_step),
                    "time left: {:5.2e}".format((num_steps - stepx - 1) * time_per_step),
                    "pe: {: 16.10e}".format(potential_energy[-1]),
                    "ke: {: 16.10e}".format(kinetic_energy[-1]),
                    "te: {: 16.10e}".format(total_energy[-1]),
                )
                sys.stdout.flush()

    particle_state.free()

    if mpi.COMM_WORLD.rank == 0:
        np.save("t.npy", np.array(list_t))
        np.save("E2.npy", np.array(list_E2))
        np.save("potential_energy.npy", np.array(potential_energy))
        np.save("kinetic_energy.npy", np.array(kinetic_energy))
        np.save("total_energy.npy", np.array(total_energy))
