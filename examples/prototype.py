from firedrake import *
from two_stream import *


def setup_project(particle_state, V):
    m = V.mesh()
    W = VectorFunctionSpace(m, V.ufl_element())
    X = assemble(interpolate(m.coordinates, W))
    quadrature_points = X.dat.data_ro
    nrow, ncol = quadrature_points.shape
    ptr = X.dat.data_ro.ctypes._as_parameter_.value
    particle_state.qpm_setup(nrow, ncol, ptr)

def project(particle_state, u):
    space = np.zeros_like(u.dat.data)
    nrow = space.shape[0]
    ncol = 1
    particle_state.qpm_project(nrow, ncol, space.ctypes._as_parameter_.value)
    u.dat.data[:] = space

if __name__ == "__main__":
    
    num_steps = 800
    num_cells = 16
    num_particles = 100000

    #mesh = PeriodicUnitSquareMesh(
    mesh = UnitSquareMesh(
        num_cells, 
        num_cells
    )
    mesh_firedrake = UnitSquareMesh(
        num_cells, 
        num_cells
    )
    
    V = FunctionSpace(mesh, "DG", 0)
    V_firedrake = FunctionSpace(mesh_firedrake, "DG", 0)
    
    u = Function(V, name="u")
    v = Function(V_firedrake)


    # Immediately pass the DM to NESO-Particles before Firedrake adds any halo
    # cells.
    particle_state = TwoStreamParticles(
        mesh.topology_dm.handle,
        num_particles,
        0.0005
    )
    particle_state.validate_halos()
    particle_state.add_particles()

    setup_project(particle_state, V)
    project(particle_state, u)

    outfile = VTKFile("mesh.pvd")
    outfile.write(mesh)

    outfile = VTKFile("u.pvd")
    outfile.write(u)
    
    particle_state.write();

    for stepx in range(num_steps):
        if (stepx % 10 == 0):
            if mpi.COMM_WORLD.rank == 0:
                print(stepx)

            project(particle_state, u)
            particle_state.write();
            outfile.write(u)

        particle_state.move()

    particle_state.free();
