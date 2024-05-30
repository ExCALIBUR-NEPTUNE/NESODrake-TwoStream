from firedrake import *
from firedrake.__future__ import interpolate
from two_stream import *


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
    
    num_steps = 20
    num_print_steps = 1
    num_cells = 16
    num_particles = 100000

    mesh_np = PeriodicUnitSquareMesh(
        num_cells, 
        num_cells
    )
    mesh = UnitSquareMesh(
        num_cells, 
        num_cells
    )

    BDM_np = FunctionSpace(mesh_np, "BDM", 1)
    DG_np = FunctionSpace(mesh_np, "DG", 0)
    interp_intermediate = Function(DG_np)

    BDM = FunctionSpace(mesh, "BDM", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    W = BDM * DG

    # Immediately pass the DM to NESO-Particles before Firedrake adds any halo
    # cells.
    particle_state = TwoStreamParticles(
        mesh.topology_dm.handle,
        num_particles,
        0.0005
    )
    
    net_charge_density = Constant(particle_state.get_net_charge_density())

    particle_state.validate_halos()
    particle_state.add_particles()

    setup_project(particle_state, DG_np)
    setup_evaluate(particle_state, DG_np)
    #project(particle_state, u)
    #evaluate(particle_state, u)
    
    E = Function(BDM)
    phi = Function(DG)

    x, y = SpatialCoordinate(mesh)
    E.interpolate(
        as_vector([cos(4*y), -sin(5*x)]),
    )
    
    project(particle_state, phi, interp_intermediate)
    evaluate(particle_state, E, interp_intermediate)

    out_phi = VTKFile("phi.pvd")
    out_phi.write(phi)
    out_E = VTKFile("E.pvd")
    out_E.write(E) 

    particle_state.write();

    particle_state.free(); quit()

    for stepx in range(num_steps):
        if (stepx % num_print_steps == 0):
            if mpi.COMM_WORLD.rank == 0:
                print(stepx)

            project(particle_state, u)
            evaluate(particle_state, u)
            particle_state.write();
            outfile.write(u)

        particle_state.move()

    particle_state.free();
