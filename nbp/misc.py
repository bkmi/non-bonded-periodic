import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import time
from numba import jit
from scipy.spatial import distance_matrix


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


class Analyser:
    """A class for the visual analysis of the nbp.System states
    Attr:
        _system (obj: <nbp.System>, private): An instance of the <nbp.System> class

        _states (obj: <list> of obj: <nbp.SystemState>, private): A list of instances of <nbp.SystemState> class

        _rdf () TODO: add description

        _energy (obj: <dict> of <list> or <array>): A dictionary containing energies of the system

    """

    def __init__(self, system=None, states=None):
        """Initialization
        Kwargs:
            system (obj: <nbp.System>): An instance of the <nbp.System> class

            states (obj: <list> of obj: <nbp.SystemState>): A list of instances of <nbp.SystemState> class
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        self._system = system
        self._states = self._system.states()
        self._rdf = None
        self._energy = {}

    def plot_distribution(self, typ=None, show=True, save=False, filename=None, fmt='png', **kwargs):
        """Distribution plotting method:
        Kwargs:
            typ (str, default=None): A string specifying the type of the distribution to be plotted

            show (bool, default=True): Boolean specifying if the plot is to be shown.

            save (bool, default=False): Boolean specifying if the generated plot is to be saved

            filename (str, optional): A string specifying the filename while saving. If none provided
            a timestamp will be used.

            fmt (str, default='png'): A string specifying the file format while saving
        """
        if not typ:
            raise ValueError("Please specify a type of distribution to be plotted")
        if typ == 'rdf':
            if not self._rdf:
                return self._get_rdf()
            raise UserWarning("Hey there! Use with caution, the feature is not yet correctly implemented!")

        if typ == 'energy':
            energy, edges = self._energy_distribution()
            plt.hist(energy, bins=edges)

        if typ == 'distances':
            return self._distances_distribution()

    def plot_energy(self, typ='total', show=True, save=False, filename=None, hline=None,  fmt='png', **kwargs):
        """Energy Plotting method:
        Kwargs:
            typ (str, default='total'): A string specifying the type of the energy to be plotted.

            show (bool, default=True): Boolean specifying if the plot is to be shown.

            save (bool, default=False): Boolean specifying if the generated plot is to be saved.

            filename (str, optional): A string specifying the filename while saving. If none provided
            a timestamp will be used.

            fmt (str, default="png"): A string specifying the file format while saving.
        """
        if typ == 'all':
            for each in ['total', 'lj', 'coulomb']:
                energy, average = self.get_energy(typ)
                figure, axes = self._setup_figure(typ, hline)
                filename = filename or "energy_{}_{}.{}".format(typ, time.strftime("%Y%M%d"), fmt)
                axes.plot(energy)
                axes.axhline(average)

                if save:
                    figure.savefig(filename)

                if show:
                    plt.show()
        else:
            energy, average = self.get_energy(typ)
            figure, axes = self._setup_figure(typ, hline)
            filename = filename or "energy_{}_{}.{}".format(typ, time.strftime("%Y%M%d"), fmt)
            axes.plot(energy)
            axes.axhline(y=average, color='green')
            axes.text(1, 5, "Average: {:.3f} kJ/mol".format(average), fontsize=15)
            if save:
                figure.savefig(filename)

            if show:
                plt.show()

    def play_frames(self, start=None, end=None, dt=None):
        """a function to play generated frames in 3d"""
        pausetime = dt or 0.01
        fig, ax = self._create_figure(subplots=1, split_axes=1, axes3d=True)
        states = self._states
        sframe = None
        # if start and end:
        #     playtime = range(start, end)
        # elif start and not end:
        #     playtime = range(start, len(states))
        # elif end and not start:
        #     playtime = range(0, end)
        #     playtime = range(len(positions))
        for frame, state in enumerate(states):
            ax.set_title("frame: {}".format(frame))
            positions = state.positions()
            if sframe:
                ax.collections.remove(sframe)
            X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]
            sframe = ax.scatter(X, Y, Z)
            plt.pause(pausetime)
        plt.close()

    def get_energy(self, typ):
        """A method for getting the requested type of energy"""
        if typ == 'total' or 1:

            energy_list = list(map(lambda x: x.energy_lj(), self._states))
            print(energy_list)
            average = np.mean(energy_list)
            print(average)
            return energy_list, average
        elif typ == 'lj' or 2:
            energy_list = list(map(lambda x: x.energy_lj(), self._states))
            print(energy_list)
            average = np.mean(energy_list)
            print(average)
            return energy_list, average

        elif typ == 'coulomb' or 3:
            energy_list = list(map(lambda x: x.energy_ewald(), self._states))
            print(energy_list)
            average = np.mean(energy_list)
            print(average)
            return energy_list, average


    def _setup_figure(self, typ, hline=None, **kwargs):
        """Private method that takes care of the figure creation,
        setting the title, axes, and labels.
        Args:
            typ (str): A string representing the type of energy to be plotted

        Kwargs:
            hline (obj: <dict> of obj: <int>, <str>; optional):
            A dictionary specifying the parameters for a horizontal line

        Returns:
            figure (obj: <matplotlib.Figure>): An instance of <matplotlib.Figure> class

            ax (obj: <matplotlib.Axes>): An instance of <matplotlib.Axes> class
        """
        if typ == "total":
            title_string = 'Total'
        elif typ == "lj":
            title_string = 'Lennard-Jones'
        elif typ == "coulomb":
            title_string = 'Coulomb'
        else:
            title_string = input("Please specify the energy type")
        fig, ax = self._create_figure()
        ax.set_title('Energy {}'.format(title_string))
        ax.set_ylabel('E [kJ/mol]')
        ax.set_xlabel('State #')
        if hline:
            if "color" and "style" in hline.keys():
                try:
                    y = hline["yval"]
                    col = hline["color"]
                    style = hline["style"]
                except IndexError:
                    y = hline["yval"]
            ax.axhline(y=y, color=col, ls=style)
        return fig, ax

    def _create_figure(self, subplots=1, split_axes=1, axes3d=False, **kwargs):
        """A method for creating a matplotlib figure and axes (or subplots) for plotting
        Kwargs:
            subplots (int): An integer specifying the number of subplots to use in a figure

            split_axes (int; default=1): An integer specifying the method for splitting
            the figure into subplots. <1> Denotes horizontal split, <2> - Vertical.

        Returns:
            fig (obj: <matplotlib.Figure>): An instance of <matplotlib.Figure> class

            ax (obj: <matplotlib.Axes>): An instance of <matplotlib.Axes> class
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if axes3d:
            ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def _get_rdf(self, bins=100):
        boxlen = self._system.info().box_dim()[0]
        boxlenh = boxlen/2
        # distances = []
        dr = boxlenh/bins
        hist = [0]*(bins + 1)
        numstates = len(self._system.states())
        npart = self._system.info().num_particles()
        rdf = np.zeros(shape=(bins, 2))

        for each in self._system.states():
            R = [each.positions()[:, 0],    # x-Components
                 each.positions()[:, 1],    # y-Components
                 each.positions()[:, 2]]    # z-Components
            for i in range(npart):
                for j in range(i+1, npart):
                    rr = [R[0][i]-R[0][j], R[1][i]-R[1][j], R[2][i]-R[2][j]]  # calculate distances component-wise
                    print(rr)
                    for each in rr:         # Minimum image convention
                        each = each + boxlen if each < -boxlenh else each
                        each = each - boxlen if each >= boxlenh else each
                    rij = np.sqrt(sum(list(map(lambda x: x**2, rr))))   # distance between atom i and j
                    bin = int(np.ceil(rij/dr))
                    if (bin <= bins):
                        hist[bin] += 1
        phi = npart/(boxlen**3)
        norm = 2 * np.pi * phi * numstates * dr * npart
        for i in range(1, bins):
            rrr = (i - 0.5) * dr
            val = hist[i]/ norm / ((rrr**2) + (dr**2) / 12.0)
            rdf[i, 0], rdf[i, 1] = rrr, val

        return rdf

    def _energy_distribution(self):
        energy = self.get_energy(typ='total')
        return np.histogram(energy, bins=100)

    def _distances_distribution(self):
        pass





