import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal


def show_frame(system, frame=None):
    """a function to visualize the particles in 3d"""

    positions = system.states()
    if frame:
        positions = positions[frame]    # select a particular frame
    else:
        positions = positions[-1]       # by default show last frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.scatter(X, Y, Z)
    plt.show()


def play_frames(system, start=None, end=None, dt=None):
    """a function to play generated frames in 3d"""
    pausetime = dt or 0.01
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = system.states()
    sframe = None
    if start and end:
        playtime = range(start, end)
    elif start and not end:
        playtime = range(start, len(positions))
    elif end and not start:
        playtime = range(0, end)
    else:
        playtime = range(len(positions))
    for frame, state in enumerate(playtime):
        ax.label = "frame: {}, state: {}".format(frame, state)
        if sframe:
            ax.collections.remove(sframe)
        X, Y, Z = positions[frame][:, 0], positions[frame][:, 1], positions[frame][:, 2]
        sframe = ax.scatter(X, Y, Z)
        plt.pause(pausetime)
    plt.close()


def wrap_pbc(positions, box_length):
    """a function that wraps particles into a defined periodic box"""
    positions[np.nonzero(positions >= box_length)] -= box_length
    positions[np.nonzero(positions < 0)] += box_length
    return positions


def generate_positions(num_particles=40, box=[1, 1, 1], pbc=True):
    """A function to generate random particle positions for a simulation system"""
    if not np.all(box) == box[0]:
        raise ValueError("A box with dimensions {}x{}x{} is not quadratic! Please chose a quadratic box.".format(box[0],
                                                                                                                 box[1],
                                                                                                                 box[2])
                         )
    box_len = box[0]
    positions = np.array([multivariate_normal(box).rvs().tolist() for _ in range(num_particles)])
    if pbc:
        positions = wrap_pbc(positions, box_len)
    return positions
