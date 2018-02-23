import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal


class Analyser:
    """A class for the visual analysis of the nbp.System states"""

    def __init__(self, system=None):
        """Initialization

        :param system: (obj: nbp.System)
            An nbp.System class instance
        """
        # if not isinstance(system, nbp.System):
        #     raise TypeError("system argument has to be an instance of nbp.System class; you provided {}".format(type(system)))


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        self._system = system
        self._states = self._system.states()
        self._rdf = None
        self._energy_lj = None
        self._energy_coul = None
        self._energy_tot = None

    def plot_distribution(self, typ=None, show=True, save=False, filename=None, fmt='png', **kwargs):
        """Distribution plotting method:

        :param typ: (str)
            A string specifying the type of distribution to be plotted
            options:
            <rdf> radial distribution function
            <energy> energy distribution
            <distances> distances distribution

        :param show: (bool)
            A boolean that switches whether the plot is to be displayed.
            default: True.

        :param save: (bool)
            A boolean that switches whether the plot is to be saved.
            default: False.

        :param filename: (str)
            A string specifying the user given name for saving the file.
            If none provided a timestamp + energy type will be used as default.

        :param fmt: (str)
            A string specifying the format of the saved file.
            default: png
        """
        if not typ:
            raise ValueError("Please specify a type of distribution to be plotted")
        if typ == 'rdf':
            if not self._rdf:
                self._rdf = self._get_rdf()
                fig, axes = self._create_figure()
                axes.set_xlabel("Radius [nm]")
                axes.set_title("Radial Distribution Function")
                axes.plot(self._rdf[:,0], self.rdf[:,1])
                if show:
                    plt.show()
                if save:
                    filename = filename or "{}_{}.{}".format(
                        typ, time.strftime("%Y%M%d"), fmt
                    )
                    fig.savefig(filename)


        if typ == 'energy':
            energy, edges = self._energy_distribution()
            plt.hist(energy, bins=edges)

        if typ == 'distances':
            return self._distances_distribution()

    def plot_energy(self, typ='total', show=True, save=False, filename=None, fmt='png', **kwargs):
        """Energy Plotting method:

        :param typ: (str)
            A string specifying the type of energy to be plotted
            options:
            <total> total energy due to particle interaction
            <lj> energy due to Van der Waals interaction
            <coulomb> energy due to coulomb interaction
            default: <total>

        :param show: (bool)
            A boolean that switches whether the plot is to be displayed.
            default: True.

        :param save: (bool)
            A boolean that switches whether the plot is to be saved.
            default: False.

        :param filename: (str)
            A string specifying the user given name for saving the file.
            If none provided a timestamp + energy type will be used as default.

        :param fmt: (str)
            A string specifying the format of the saved file.
            default: png
        """
        if typ == "total":
            title_string = 'Total'
        elif typ == "lj":
            title_string = 'Lennard-Jones'
        elif typ == "coulomb":
            title_string = 'Coulomb'
        else:
            title_string = input("Please specify the energy type")
        energy, average = self.get_energy(typ)
        figure, axes = self._create_figure(axes3d=False)
        filename = filename or "energy_{}_{}.{}".format(typ, time.strftime("%Y%M%d"), fmt)
        axes.plot(energy)
        axes.set_title('Energy {}'.format(title_string))
        axes.set_ylabel('E [kJ/mol]')
        axes.set_xlabel('State #')
        axes.axhline(y=average, color='green')
        axes.text(1, 5, "Average: {:.3f} kJ/mol".format(average), fontsize=15)
        if save:
            figure.savefig(filename)
        if show:
            plt.show()

    def play_frames(self, dt=None):
        """A method to play generated frames in 3d

        :param dt: (float)
            A float specifying pause time between played frames
        """
        pausetime = dt or 0.01
        fig, ax = self._create_figure(subplots=1, split_axes=1, axes3d=True)
        states = self._states
        sframe = None
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
        """A method for getting the requested type of energy

        :param typ: (string)
            A string specifying the type of energy to fetch
            options:
                <total>     Total energy
                <coulomb>   Coulomb interaction energy
                <lj>        Wan der Waals interraction energy

        :returns (energy_list, average): (list of float, float)
            A list containing energies for each state
            A float for the average energy
        """
        if typ == 'total' or 1:
            energy_list = list(map(lambda x: x.energy(), self._states))
            average = np.mean(energy_list)
            return energy_list, average
        elif typ == 'lj' or 2:
            energy_list = list(map(lambda x: x.energy_lj(), self._states))
            average = np.mean(energy_list)
            return energy_list, average
        elif typ == 'coulomb' or 3:
            energy_list = list(map(lambda x: x.energy_ewald(), self._states))
            average = np.mean(energy_list)
            return energy_list, average

    def _create_figure(self, axes3d=False, **kwargs):
        """A method for creating a matplotlib figure and axes (or subplots) for plotting

        :param subplots: (int)
            Number of subplots
            default: 1

        :param axes3d: (bool)
            A boolean switching whether to plot in 3 dimensions
            default: False

        :returns (fig, ax): (obj, obj)
            A matplotlib.Figue and a matplotlib.Axes instance
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if axes3d:
            ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def _get_rdf(self, bins=300):
        boxlen = self._system.info().box[0]
        boxlenh = boxlen/2
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
                    # print(rr)
                    for each in rr:         # Minimum image convention
                        each = each + boxlen if each < -boxlenh else each
                        each = each - boxlen if each >= boxlenh else each
                    rij = np.sqrt(sum(list(map(lambda x: x**2, rr))))   # distance between atom i and j
                    bin = int(np.ceil(rij/dr))
                    if (bin <= bins):
                        hist[bin] += 1
        phi = npart/(boxlen**3)
        norm = (4 / 3) * np.pi * phi * numstates * dr * npart
        for i in range(1, bins):
            rrr = (i - 0.1) * dr
            val = hist[i] / (norm) / (rrr + dr)**3
            rdf[i, 0], rdf[i, 1] = rrr, val
        return rdf

    def _energy_distribution(self):
        energy = self.get_energy(typ='total')
        return np.histogram(energy, bins=100)

    def _distances_distribution(self):
        pass

