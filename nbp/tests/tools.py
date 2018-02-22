import nbp
import numpy as np


def make_system(characteristic_length=10,
                sigma=None, epsilon_lj=None, particle_charges=None, positions=None, particle_count=None,
                lj=True, ewald=True, use_neighbours=False, reci_cutoff=5):
    if particle_count is None:
        if particle_charges is not None:
            particle_count = np.asarray(particle_charges).shape[0]
        elif positions is not None:
            particle_count = np.asarray(positions).shape[0]
        else:
            particle_count = 50

    if not sigma:
        sigma = np.ones((particle_count, 1))
    if not epsilon_lj:
        epsilon_lj = np.ones((particle_count, 1))

    if particle_charges is None:
        particle_charges = np.random.rand(particle_count, 1)
    if positions is None:
        positions = characteristic_length * np.random.rand(particle_count, 3)

    system = nbp.System(characteristic_length, sigma, epsilon_lj, particle_charges, positions,
                        reci_cutoff=reci_cutoff, lj=lj, ewald=ewald, use_neighbours=use_neighbours)
    return system
