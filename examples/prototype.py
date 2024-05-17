from firedrake import *
from two_stream import *

if __name__ == "__main__":

    num_cells = 5
    num_particles = 1000

    mesh = UnitSquareMesh(
        num_cells, 
        num_cells
    )
    # Immediately pass the DM to NESO-Particles before Firedrake adds any halo
    # cells.
    mesh.topology_dm.incRef()
    particle_state = TwoStreamParticles(
        mesh.topology_dm.handle,
        num_particles,
        0.001
    )

    particle_state.validate_halos()

    outfile = VTKFile("mesh.pvd")
    outfile.write(mesh)

    particle_state.validate_halos()





    particle_state.free();
    mesh.topology_dm.decRef()
