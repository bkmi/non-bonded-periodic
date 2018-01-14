from . import SystemInfo, SystemState
from . import Neighbours
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
    neighbours_list = neighbours.get_neighbours(positions[78])

    return neighbours_list


def show_frame(coordinates):
    """a function to visualize the particles in 3d"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0],
               coordinates[:, 1],
               coordinates[:, 2], )
    plt.show()


nb_result = test_neighbours()

print("position result:", nb_result.nb_pos)
print("distance result:",  nb_result.nb_dist)

#show_frame(coord)
