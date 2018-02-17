import nbp
import numpy as np

from nbp.tests.tools import make_system


@nbp.timing
def setup(specific_pos=False, use_neighbours=False):
    characteristic_length = 20
    if specific_pos:
        positions = (np.asarray([[1, 0, -2 ** (-1 / 2)],
                                 [-1, 0, -2 ** (-1 / 2)],
                                 [0, 1, 2 ** (-1 / 2)],
                                 [0, -1, 2 ** (-1 / 2)]]) + characteristic_length / 2) * 3
    else:
        positions = np.random.rand(4, 3) * characteristic_length

    system = make_system(characteristic_length=characteristic_length, positions=positions, lj=False, ewald=True,
                         use_neighbours=use_neighbours, reci_cutoff=5)
    return system


@nbp.timing
def optimize(system, cov):
    system = system.optimize(cov=cov, num_particles=2)
    print(len(system.states()))
    print(system.state().distance().distances_unwrapped())

    return system


@nbp.timing
def simu(system, steps, temp):
    system.simulate(steps, temp)
    print(len(system.states()))

    return system


# For no neighbour list
# sys = setup()
# op_sys = optimize(sys, sys.info().cutoff() / 24)
# op_sys = optimize(op_sys, sys.info().cutoff() / 32)
# simu_sys = simu(op_sys, 100, 100)

# With the neighbour list
sys = setup(use_neighbours=True)
op_sys = optimize(sys, sys.info().cutoff() / 24)
op_sys = optimize(op_sys, sys.info().cutoff() / 32)
simu_sys = simu(op_sys, 100, 100)
