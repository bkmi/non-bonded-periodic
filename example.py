import nbp
import numpy as np

characteristic_length = 10
sigma = 2
particle_charges = np.asarray([1, 1])
positions = np.asarray([[1, 1, 1], [2,2,2]])

a = nbp.System(characteristic_length, sigma, particle_charges, positions)
a.state().energy()