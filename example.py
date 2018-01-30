import nbp
import numpy as np

from nbp.tests.tools import make_system

# Example using LJ potential only
# characteristic_length = 10
# particle_count = 4
# system = make_system(characteristic_length=characteristic_length,
#                      particle_count=particle_count,
#                      lj=True,
#                      ewald=False,
#                      use_neighbours=False)
# system.optimize()

# blah

# np.random.RandomState(6)
characteristic_length = 10
positions = (np.asarray([[1, 0, -2**(-1/2)],
                        [-1, 0, -2**(-1/2)],
                        [0, 1, 2**(-1/2)],
                        [0, -1, 2**(-1/2)]]) + 5) * 1.5
system = make_system(characteristic_length=characteristic_length,
                     positions=positions,
                     lj=True,
                     ewald=False,
                     use_neighbours=False)
system.optimize(num_particles=4)
print(len(system.states()))
print(system.state().distances_unwrapped())
print(system.state().distances_wrapped())
print(system.state().positions())
