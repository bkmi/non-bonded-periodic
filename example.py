import nbp
import numpy as np

# Example using LJ potential only1

characteristic_length = 20
sigma = 0
particle_charges = np.asarray([1, 1, 1])
positions = np.asarray([[1, 1, 1], [2,2,2], [4,5,10]])

a = nbp.System(characteristic_length, sigma, particle_charges, positions)
a.state().energy()
print(a.state().energy())
