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


class Analyser:
    """A class for the Analysis of the system states"""

    def __init__(self, system):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        self._system = system
        self._states = self._system.states()
        self._rdf = None

    def plot_distribution(self, typ=None, save=False, fmt=None, **kwargs):
        """A method for plotting distributions"""
        if not typ:
            raise ValueError("Please specify a type of distribution to be plotted")
        if typ == 'rdf':
            if not self._rdf:
                pass

    def plot_energy(self, typ='total', save=False, fmt=None, **kwargs):
        """Energy Plotting method
        :param typ (str or int)
            type of energy to plot;
            'total' or 1 for total energy
            'lj' or 2 for Lennard Jones energy
            'coulomb' or 3 for Coulomb Energy
        """
        energy = self._get_energy(typ)
        figure, axes = self._setup_figure()



    def _get_energy(self, typ):
        """private method for getting the requested type of energy"""
        if typ == 'total' or 1:
            return list(map(lambda x: x.energy))
        elif typ == 'lj' or 2:
            return list(map(lambda x: x.energy_lj))
            plt.show()
        elif typ == 'coulomb' or 3:
            return list(map(lambda x: x.energy_ewald))
            plt.show()

    def _setup_figure(*args, **kwargs):
        """Private method that takes care of the figure creation, setting the title, axes, and labels"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return fig, ax

    def _plot_e_clmb(self):
        pass

    def _plot_e_lj(self):
        pass

    def _plot_e_tot(self):
        pass

    #TODO: add scaling for the axis

    def _rdf(self):
        pass

    def _distance_distr(self):
        pass






