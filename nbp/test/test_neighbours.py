from sysmodule import SystemInfo, SystemState
from neighbours import Neighbours
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def test_neighbours():
    # random positions and charges (charges are not needed here)
    positions = 0.05 * np.random.random_sample((100000, 3))
    charges = np.random.random_sample((100000, 1))
    # random char length and sigma
    system_info = SystemInfo(1, 0.001, charges)
    system_state = SystemState(positions)
    neighbours = Neighbours(system_info, system_state)
    nb_list = neighbours.get_neighbours(positions[2039])

    return nb_list


def show_frame(coordinates):
    """a function to visualize the particles in 3d"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0],
               coordinates[:, 1],
               coordinates[:, 2], )
    plt.show()


nb_list = test_neighbours()

print(nb_list)
#show_frame(coord)
