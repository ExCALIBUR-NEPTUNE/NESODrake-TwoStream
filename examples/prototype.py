from firedrake import *
from two_stream import *

if __name__ == "__main__":
    
    num_steps = 200
    num_cells = 32
    num_particles = 10

    mesh = PeriodicUnitSquareMesh(
        num_cells, 
        num_cells
    )

    # Immediately pass the DM to NESO-Particles before Firedrake adds any halo
    # cells.
    particle_state = TwoStreamParticles(
        mesh.topology_dm.handle,
        num_particles,
        0.001
    )
    particle_state.validate_halos()

    outfile = VTKFile("mesh.pvd")
    outfile.write(mesh)

    for stepx in range(num_steps):
        if (stepx % 10 == 0):
            if mpi.COMM_WORLD.rank == 0:
                print(stepx)
            particle_state.write();
        particle_state.move()


    particle_state.free();
