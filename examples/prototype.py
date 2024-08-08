from firedrake import *
from firedrake.__future__ import interpolate
from two_stream import *
import sys
import time

def setup_project(particle_state, V):
    m = V.mesh()
    W = VectorFunctionSpace(m, V.ufl_element())
    X = assemble(interpolate(m.coordinates, W))
    quadrature_points = X.dat.data_ro
    nrow, ncol = quadrature_points.shape
    ptr = X.dat.data_ro.ctypes._as_parameter_.value
    particle_state.setup_project(nrow, ncol, ptr)

def setup_evaluate(particle_state, V):
    m = V.mesh()
    W = VectorFunctionSpace(m, V.ufl_element())
    X = assemble(interpolate(m.coordinates, W))
    quadrature_points = X.dat.data_ro
    nrow, ncol = quadrature_points.shape
    ptr = X.dat.data_ro.ctypes._as_parameter_.value
    particle_state.setup_evaluate(nrow, ncol, ptr)

def project(particle_state, v, u):
    space = np.zeros_like(u.dat.data)
    nrow = space.shape[0]
    ncol = 1
    particle_state.project(nrow, ncol, space.ctypes._as_parameter_.value)
    u.dat.data[:] = space
    v.interpolate(u)

def evaluate(particle_state, u, v):
    for cx in range(2):
        v.interpolate(u[cx])
        space = v.dat.data_ro
        nrow = space.shape[0]
        ncol = 1
        particle_state.evaluate(cx, nrow, ncol, space.ctypes._as_parameter_.value)


if __name__ == "__main__":
    
    num_steps = 100
    num_print_steps = 5
    num_write_steps = 5
    num_energy_steps = 2

    num_cells_y = 3
    num_cells_x = 100
    num_particles = 400000
    dt = 0.001
    p = 1

    mesh_width = 0.01

    mesh = PeriodicRectangleMesh(
        num_cells_x, 
        num_cells_y,
        1.0,
        mesh_width
    )
    mesh_np = RectangleMesh(
        num_cells_x, 
        num_cells_y,
        1.0,
        mesh_width
    )

    BDM_np = FunctionSpace(mesh_np, "BDM", p+1)
    DG_np = FunctionSpace(mesh_np, "DG", p)
    interp_intermediate = Function(DG_np)

    BDM = FunctionSpace(mesh, "BDM", p+1)
    DG = FunctionSpace(mesh, "DG", p)
    W = BDM * DG

    # Immediately pass the DM to NESO-Particles before Firedrake adds any halo
    # cells.
    particle_state = TwoStreamParticles(
        mesh_np.topology_dm.handle,
        num_particles,
        dt,
        p
    )
    particle_state.validate_halos()

    particle_state.add_particles()
    
    setup_project(particle_state, DG_np)
    setup_evaluate(particle_state, DG_np)
    
    E = Function(BDM)
    rho = Function(DG, name="rho")
    neutralising_field = Function(DG, name="neutralising_field")
    poisson_rhs = Function(DG, name="poisson_rhs")
    net_charge_density = particle_state.get_net_charge_density()
    neutralising_field.interpolate(net_charge_density)
    
    poisson_rhs.interpolate(neutralising_field - rho)
    
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = poisson_rhs*v*dx
    
    w = Function(W)
    E, phi = w.subfunctions
    
    project(particle_state, rho, interp_intermediate)
    evaluate(particle_state, E, interp_intermediate)
    poisson_rhs.interpolate(neutralising_field - rho)

    rho_integral = assemble(rho * dx)
    rhs_integral = assemble(poisson_rhs * dx)
    
    print("rho_integral:", rho_integral)
    print("rhs_integral:", rhs_integral)
    
    out_rho = VTKFile("rho.pvd")
    out_E = VTKFile("E.pvd")
    
    list_t = []
    list_E2 = []

    t0 = time.time()
    for stepx in range(num_steps):
        project(particle_state, rho, interp_intermediate)
        poisson_rhs.interpolate(neutralising_field - rho)
        solve(a == L, w, bcs=[])
        evaluate(particle_state, E, interp_intermediate)

        if (stepx % num_write_steps == 0) and (num_write_steps > 0):
            out_rho.write(rho, poisson_rhs)
            out_E.write(E) 
            particle_state.write();
    
        particle_state.move()

        if (stepx % num_energy_steps == 0):
            list_t.append(stepx * dt)
            list_E2.append(norm(E) ** 2.0)

        if (stepx % num_print_steps == 0):
            if mpi.COMM_WORLD.rank == 0:
                time_per_step = (time.time() - t0) / (stepx+1)
                print(
                    "step:", stepx,
                    "time per step: {:5.2e}".format(time_per_step), 
                    "time left: {:5.2e}".format((num_steps - stepx - 1) * time_per_step)
                )
                sys.stdout.flush()

    particle_state.free();

    if mpi.COMM_WORLD.rank == 0:
        array_t = np.array(list_t)
        array_E2 = np.array(list_E2)
        np.save("t.npy", array_t)
        np.save("E2.npy", array_E2)

