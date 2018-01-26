import nbp
import numpy as np

def make_system(lj=True, ewald=True, use_neighbours=False):
    particle_count = 50
    sigma = np.ones(particle_count, 1)
    epsilon_lj = np.ones(particle_count, 1)

    box_length = 10
    random_positions = box_length * np.random.rand(particle_count, 3)
    random_charges = np.random.rand(particle_count, 1)

    system = nbp.System(box_length, sigma, epsilon_lj, random_charges, random_positions,
                        lj = lj, ewald = ewald, use_neighbours = use_neighbours)
    return system