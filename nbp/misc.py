import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal


def show_frame(system, frame=None):
    """a function to visualize the particles in 3d"""

    states = system.states()
    if frame:
        positions = states[frame].positions()    # select a particular frame
    else:
        positions = states[-1].positions()     # by default show last frame
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
    states = system.states()
    sframe = None
    # if start and end:
    #     playtime = range(start, end)
    # elif start and not end:
    #     playtime = range(start, len(states))
    # elif end and not start:
    #     playtime = range(0, end)
    # else:
    #     playtime = range(len(positions))
    for frame, state in enumerate(states):
        ax.label = "frame: {}".format(frame, state)
        positions = state.positions()
        if sframe:
            ax.collections.remove(sframe)
        X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        sframe = ax.scatter(X, Y, Z)
        plt.pause(pausetime)
    plt.close()

