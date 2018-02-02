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


class Plotter:
    """A class for plotting energies"""
    def __init__(self, system):
        import matplotlib.pyplot as plt
        self._system = system
        self._states = self._system.states()
        self._energies = None

    def plot_distribution(self):
        pass

    def plot_energies(self, typ='total'):
        """Energy Plotting method
        :param typ (str or int)
            type of energy to plot;
            'total' or 1 for total energy
            'lj' or 2 for Lennard Jones energy
            'coulomb' or 3 for Coulomb Energy
        """
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        self.get_energies()
        if typ == 'total' or 1:
            self._plot_e_tot(ax)
            ax.label = 'Total Energy'
            plt.show()
        elif typ == 'lj' or 2:
            self._plot_e_lj(ax)
            ax.label = 'Energy LJ'#
            plt.show()
        elif typ == 'coulomb' or 3:
            self._plot_e_clmb(ax)
            ax.label = 'Energy Coulomb'
            plt.show()

    def get_energies(self):
        if self._energies == None:
            self._energies = dict()
            self._energies['total'] = list(map(lambda x: x.energy(), self._states))
            self._energies['lj'] = list(map(lambda x: x.energy_lj(), self._states))
            self._energies['coulomb'] = list(map(lambda x: x.energy_ewald(), self._states))

    def _plot_e_clmb(self, canvas):
        canvas.plot(self._energies['coulomb'])

    def _plot_e_lj(self, canvas):
        canvas.plot(self._energies['lj'])

    def _plot_e_tot(self, canvas):
        canvas.plot(self._energies['total'])

    #TODO: add scaling for the axis

    def _rdf(self):
        pass

    def _distance_distr(self):
        pass






