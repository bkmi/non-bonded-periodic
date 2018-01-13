from sysmodule import SystemInfo, SystemState
from neighbours import Neighbours
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def test_neighbours():
    # random positions and charges (charges are not needed here)
    positions = 4 * np.random.random_sample((10000, 3))
    charges = np.random.random_sample((100, 1))
    # random char length and sigma
    system_info = SystemInfo(6, 0.2, charges)
    system_state = SystemState(positions)
    neighbours = Neighbours(system_info, system_state)
    nb_list = neighbours.get_neighbours(positions[78])
    nb_dist = neighbours.get_neighbours_distance(positions[78])

    return nb_list, nb_dist


def show_frame(coordinates):
    """a function to visualize the particles in 3d"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0],
               coordinates[:, 1],
               coordinates[:, 2], )
    plt.show()


nb_list = test_neighbours()[0]
nb_dist = test_neighbours()[1]

print("result:", nb_list)
print("result distances:", nb_dist)
#show_frame(coord)
