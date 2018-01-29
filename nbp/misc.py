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

class EnergyPlotter:
    """A class for plotting energies"""
    def __init__(self, system):
        import matplotlib.pyplot as plt
        self._system = system
        self._states = self._system.states()
        self._energies = None


    def get_energies(self):
        if self._energies == None:
            self._energies = dict()
            self._energies['total'] = list(map(lambda x: x.energy(), self._states))
            self._energies['lj'] = list(map(lambda x: x.energy_lj(), self._states))
            self._energies['coulomb'] = list(map(lambda x: x.energy_ewald(), self._states))

    def _plot_e_clmb(self, canvas):
        canvas.plot(self._energies['coulomb'])
        plt.show()

    def _plot_e_lj(self, canvas):
        canvas.plot(self._energies['lj'])
        plt.show()

    def _plot_e_tot(self, canvas):
        canvas.plot(self._energies['total'])
        plt.show()

    def plot_energies(self, typ='total', start=0, end=None):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        if end == None:
            end = len(self._states)
        self.get_energies()
        if typ == 'total':
            ax.label = 'Total Energy'
            self._plot_e_tot(ax)
        elif typ == 'lj':
            ax.label = 'Energy LJ'
            self._plot_e_lj(ax)
        elif typ == 'coulomb':
            ax.label = 'Energy Coulomb'
            self._plot_e_clmb(ax)



