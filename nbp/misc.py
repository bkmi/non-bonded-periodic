import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_frame(coordinates):
    """a function to visualize the particles in 3d"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0],
               coordinates[:, 1],
               coordinates[:, 2], )
    plt.show()