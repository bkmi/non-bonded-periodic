import nbp

characteristic_length = 10
sigma = 2
particle_charges = [1, 1]
positions = [[1, 1, 1], [2,2,2]]

a = nbp.System(characteristic_length, sigma, particle_charges, positions)
