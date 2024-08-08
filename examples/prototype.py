from firedrake import *
from firedrake.__future__ import interpolate
from two_stream import *
import sys
import time

class ProjectEvaluate:
    def __init__(self, particle_state, V):
        self.particle_state = particle_state
        self.intermediate_func = Function(V)

        m = V.mesh()
        W = VectorFunctionSpace(m, V.ufl_element())
        X = assemble(interpolate(m.coordinates, W))
        quadrature_points = X.dat.data_ro
        nrow, ncol = quadrature_points.shape
        ptr = X.dat.data_ro.ctypes._as_parameter_.value
        particle_state.setup_project(nrow, ncol, ptr)

    def evaluate(self, u, sym_name):
        self.intermediate_func.interpolate(u)
        space = self.intermediate_func.dat.data_ro
        nrow = space.shape[0]
        ncol = 1
        particle_state.evaluate(sym_name, nrow, ncol, space.ctypes._as_parameter_.value)

    def project(self, sym_name, u):
        space = self.intermediate_func.dat.data
        nrow = space.shape[0]
        ncol = 1
        particle_state.project(sym_name, nrow, ncol, space.ctypes._as_parameter_.value)
        u.interpolate(self.intermediate_func)


if __name__ == "__main__":
    
    num_steps = 100
    num_print_steps = 1
    num_write_steps = 5
    num_energy_steps = 1

    num_cells_y = 3
    num_cells_x = 100
    num_particles = 400000
    dt = 0.001
    p = 1
    mesh_width = 0.01

    mesh_np = RectangleMesh(
        num_cells_x, 
        num_cells_y,
        1.0,
        mesh_width
    )
    DG_np = FunctionSpace(mesh_np, "DG", p)
    particle_state = TwoStreamParticles(
        mesh_np.topology_dm.handle,
        num_particles,
        dt,
        p
    )
    pe = ProjectEvaluate(particle_state, DG_np)



    mesh = PeriodicRectangleMesh(
        num_cells_x, 
        num_cells_y,
        1.0,
        mesh_width
    )
    BDM = FunctionSpace(mesh, "BDM", p+1)
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
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = poisson_rhs*v*dx

    w = Function(W)
    E, phi = w.subfunctions
    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True)]
    )
    
    pe.project("Q", rho)

    poisson_rhs.interpolate(neutralising_field - rho)

    rho_integral = assemble(rho * dx)
    rhs_integral = assemble(poisson_rhs * dx)
    
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

    t0 = time.time()
    for stepx in range(num_steps):
        pe.project("Q", rho)
        poisson_rhs.interpolate(neutralising_field - rho)
        solve(a == L, w, bcs=[], nullspace=nullspace)
        pe.evaluate(E[0], "E0")
        pe.evaluate(E[1], "E1")
        pe.evaluate(phi, "PHI")

        if (stepx % num_write_steps == 0) and (num_write_steps > 0):
            out_rho.write(rho, poisson_rhs)
            out_E.write(E) 
            particle_state.write();
    
        particle_state.move()
        
        last_potential_energy = 0.0
        last_kinetic_energy = 0.0

        if (stepx % num_energy_steps == 0):
            list_t.append(stepx * dt)
            list_E2.append(norm(E) ** 2.0)
            update_particle_energy()

        if (stepx % num_print_steps == 0):
            if mpi.COMM_WORLD.rank == 0:
                time_per_step = (time.time() - t0) / (stepx+1)
                print(
                    "step:", stepx,
                    "time per step: {:5.2e}".format(time_per_step), 
                    "time left: {:5.2e}".format((num_steps - stepx - 1) * time_per_step),
                    "pe: {:16.10e}".format(potential_energy[-1]),
                    "ke: {:16.10e}".format(kinetic_energy[-1]),
                    "te: {:16.10e}".format(total_energy[-1]),
                )
                sys.stdout.flush()

    particle_state.free();

    if mpi.COMM_WORLD.rank == 0:
        array_t = np.array(list_t)
        array_E2 = np.array(list_E2)
        np.save("t.npy", array_t)
        np.save("E2.npy", array_E2)

