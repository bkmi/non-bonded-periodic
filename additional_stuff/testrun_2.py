from nbp.misc import Analyser
from nbp.markov import *
from nbp.distance import *
from scipy.spatial import distance_matrix
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import nbp

class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self._system = system

    def optimize(self, max_steps=500, cov=None, d_energy_tol=1e-6, no_progress_break=250, num_particles=0.25):
        """Optimize from the last system state."""
        optimizer = Optimizer(self._system)
        energies = []
        if cov is None:
            cov = self._system.info.cutoff/6
        for i in range(max_steps):
            new_state, new_energy = optimizer.act(cov, num_particles=num_particles)
            self._system.update_state(new_state)
            energies.append(new_energy)
            if len(energies) > no_progress_break and np.all(
                    np.less(np.abs(np.asarray(energies)[-no_progress_break:] - new_energy), d_energy_tol)):
                break
        return self._system.state

    def simulate(self, steps, temperature):
        """Simulate from the last system state."""
        simulator = Simulator(self._system)
        for i in range(steps):
            # print("step:", i)
            self._system.update_state(simulator.act(temperature))
        print("accepted states number: ", simulator.accepted_number)
        return self._system.state


class Optimizer:
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system):
        self._system = system
        self._proposal = None

    def _propose(self, cov, num_particles=None):
        """Propose the next state, moves a num_particles randomly with a 3d gaussian.

        returns proposal state, proposal_energy"""
        positions = self._system.state.positions
        if isinstance(num_particles, float) and num_particles <= 1:
            num_particles = int(positions.shape[0] * num_particles)
        elif isinstance(num_particles, int) and num_particles <= positions.shape[0]:
            pass
        else:
            raise ValueError('num_particles must be a percentage (float) or a number of particles (int).')
        particles = np.random.choice(positions.shape[0], size=num_particles, replace=False)
        proposal_positions = positions
        proposal_positions[particles] = positions[particles] + sp.stats.multivariate_normal(
            np.zeros(3), cov * np.eye(3)).rvs(num_particles)
        proposal_state = SystemState(self._system, proposal_positions)
        return proposal_state, proposal_state.energy_lj

    @staticmethod
    def _check(orig_energy, proposal_energy):
        if proposal_energy <= orig_energy:
            return True
        else:
            return False

    def act(self, cov, num_particles=0.25):
        """Propose and check then return a new state."""
        orig_energy = self._system.state.energy_lj
        self._proposal, proposal_energy = self._propose(cov, num_particles=num_particles)
        if self._check(orig_energy, proposal_energy):
            return self._proposal, proposal_energy
        else:
            return self._system.state, orig_energy


class Simulator:
    """The class that simulates."""
    def __init__(self, system):
        self._system = system
        self._accepted_number = 0

    def act(self, temperature):
        """
        A method for returning proposal states in the MCMC
            :parameter: temperature (float)
                temperature in Kelvin [K]
        """
        cov = self._system.info().sigma_lj/20
        num_particles = len(self._system.state().positions())
        indices_toMove = list(set(np.random.choice(np.arange(num_particles), size=int(np.ceil((0.25*num_particles))))))
        proposal_state = self._metropolis(indices_toMove, cov)
        if self._check(proposal_state, temperature):
            self._accepted_number += 1
            return proposal_state
        else:
            return self._system.state()

    def _check(self, state, temperature):
        """
        A method for checking for the acceptance of the proposed state
            :parameter: state (obj)
            :parameter: temperature (float)
        """
        beta = 1
        energy_prev = self._system.state().energy_lj()
        energy_prop = state.energy_lj()
        p_acc = np.min((1, np.exp(-beta * (energy_prop - energy_prev))))
        random_number = np.random.random()
        if random_number <= p_acc:
            return True
        else:
            return False

    def _metropolis(self, indices, cov):
        """Proposes the new states"""
        new_positions = np.copy(self._system.state().positions())
        new_positions[indices] += np.array(
            [sp.stats.multivariate_normal(np.zeros(3), cov=cov).rvs().tolist() for each in new_positions[indices]])
        proposal_state = SystemState(self._system, new_positions)
        return proposal_state

    @property
    def accepted_number(self):
        return self._accepted_number



class System:
    def __init__(self, parameters, positions=None, **kwargs):
        self._ipar = parameters
        self._systemInfo = SystemInfo(self)
        self._ipos = positions #or self.generate_positions(self._ipar['num_particles'],
                                                          # self._ipar['box'],
                                                          # self._ipar['positions']
                                                          # )
        self._MCMC = MCMC(self)
        self._states = [SystemState(self, self._ipos)]

    @staticmethod
    def generate_positions(num, box, method):
        """A method for generating initial positions

            Parameters:
            -----------

            num:    (integer)
                    Number of particles

            box:    (list or array-like)
                    A list containing box dimensions

            method: (string)
                    A string specifying a method for
                    generating initial positions
                    Options: <random> or <grid> (Not implemented)

            Returns:
            --------

            positions:  (numpy-array)
                        A (num x 3)-array containing
                        particle coordinates

            Example:
            --------
            initial_positions = generate_positions(100, [20, 20, 20], "random")

            initial_positions is a (100 x 3)-array containing 100 particles' coordinates
            assigned randomly in a cubic simulation box with a side-length 20

            """
        if method == "random":
            positions = np.random.rand(num, 3)*box
        elif method == 'grid':
            raise NotImplementedError("The {} method has not been implemented yet!".format(method))
        return positions

    def update_state(self, new_state):
        if isinstance(new_state, SystemState):
            self._states.append(new_state)
        elif isinstance(new_state, list) and isinstance(new_state[0], SystemState):
            self._states.extend(new_state)
        else:
            raise TypeError('SystemState could not be updated. Item is not type(SystemState) or a list of SystemState')


    def info(self):
        return self._systemInfo

    def state(self):
        """Gives the current dynamic information about the system"""
        return self._states[-1]

    def states(self):
        """Gives all the dynamic information about the system"""
        return self._states

    def optimize(self, *args, **kwargs):
        """Optimize the system to a lower energy level."""
        return self._MCMC.optimize(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """Simulate the system at a given temperature"""
        return self._MCMC.simulate(*args, **kwargs)


    def extend_states(self, states):
        self._states = self._states.extend(states)
        # return self._states


class SystemInfo:

    def __init__(self, system):
        self._system = system
        self._box = self._system._ipar['box']
        self._sigma_lj = self._system._ipar['sigma_lj']
        self._epsilon_lj = self._system._ipar['epsilon_lj']
        self._cutoff_radius = self._sigma_lj * 2.5
        self._sigma_eff = None
        self._particle_types = self._system._ipar['types']
        self._particle_charges = self._system._ipar['charges']
        self._pbc = self._system._ipar['pbc']

    def system(self):
        return self._system

    def num_particles(self):
        return len(self.system().state().positions())

    @property
    def box(self):
        return self._box

    @property
    def volume(self):
        return self._box[0]*self._box[1]*self._box[2]

    @property
    def cutoff(self):
        """Returns the value chosen for the cutoff radius"""
        return self._cutoff_radius

    @property
    def sigma_eff(self):
        return self._sigma_eff

    @property
    def sigma_lj(self):
        return self._sigma_lj

    @property
    def epsilon_lj(self):
        return self._epsilon_lj

    @property
    def particle_charges(self):
        return self._particle_charges

    @property
    def pbc(self):
        return self._pbc


class SystemState:

    def __init__(self, system, positions, verbose=False):
        self._system = system
        self._positions = positions
        self._energy_lj = None
        self._energy_cou = None
        self._energy_tot = None
        self._neighbours = None
        self._distance = None

    def system(self):
        return self._system


    def wrap_distances(self, positions):
        npart = len(positions)
        distance_matr = np.zeros(shape=(npart, npart))
        R = [positions[:, 0], positions[:, 1], positions[:, 2]]
        dR = self.system().info().box[0]
        dr = dR/2
        for i in range(npart):
            for j in range(i+1, npart):
                rr = [R[0][i]-R[0][j], R[1][i]-R[1][j], R[2][i]-R[2][j]]  # calculate distances component-wise
                for component in rr:         # Minimum image convention
                    component = component + dR if component < -dr else component
                    component = component - dR if component >= dr else component
                distance_matr[i, j] = np.sqrt(sum(list(map(lambda x: x**2, rr))))
        return distance_matr


    def wrap_positions(self, positions):
        R = [positions[:, 0], positions[:, 1], positions[:, 2]]
        dR = [self.system().info().box[0], self.system().info().box[1], self.system().info().box[2]]
        for i, r in enumerate(R):
            r[np.nonzero(r < 0)] = r[np.nonzero(r < 0)] + dR[i]
            r[np.nonzero(r >= dR[i])] = r[np.nonzero(r >= dR[i])] - dR[i]
            positions[:, i] = r
        return positions

    @property
    def pbc(self):
        return self._system.info().pbc

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        if self.pbc:
            self._positions = self.wrap_positions(self._positions)
        return self._positions

    @property
    def distance(self):
        if self._distance is None:
            self._distance = np.triu(distance_matrix(self.positions(), self.positions()))
            # print(self._distance)
        return self._distance

    @staticmethod
    def lennard_jones(sigma, epsilon, distance):
        # print(distance)
        q = (sigma/distance)**6
        return 4*epsilon*q*(q-1)

    def energy_lj(self):
        if not self._energy_lj:
            self._energy_lj = 0
            sigma = self._system.info().sigma_lj
            epsilon = self._system.info().epsilon_lj
            positions = self.positions()
            wrapped_dm = self.wrap_distances(positions)
            self._energy_lj += np.sum(
                np.asarray(
                    list(
                        map(
                            lambda x: self.lennard_jones(sigma, epsilon, x),
                            wrapped_dm[np.nonzero(wrapped_dm > 0)])
                                )
                           )
            )
        return self._energy_lj


if __name__ == '__main__':
    positions_all = np.load('D:/Uni/Master/CompSci/lj_md/trajectory_300.npy')

    parameters = {
        'num_particles': 100,    # number of particles
        'types': None,          # types of particles ('Ar', 'Na+') / ('Na+', 'Cl-')
        'epsilon_lj': 0.9363,   # in kJ/mol
        'sigma_lj': 0.340,      # in nanometers
        'charges': None,        # charges tuple ('Ar', 0) / ('Na+', 0.345) / ('Cl-', 0.345)
        'box': [8, 8, 8],       # box dimensions
        'positions': 'random',  # method for positions generation (random or grid)
        'pbc': True
    }

    positions_start = positions_all[0]
    syst = System(parameters, positions=positions_start)

    # syst2 = nbp.System(parameters['box'][0],
    #                    np.ones(len(positions_start))*parameters['sigma_lj'],
    #                    np.ones(len(positions_start))*parameters['epsilon_lj'],
    #                    np.zeros(len(positions_start)),
    #                    positions=positions_start, ewald=False, reci_cutoff=5)
    # print(list(map(lambda x: SystemState(syst, x), positions_all)))
    # syst.extend_states(list(map(lambda x: SystemState(syst, x), positions_all)))
    # for each in positions_all:
        # syst2.update_state(nbp.SystemState(each, syst2))
        # syst.update_state(SystemState(syst, each))
    # syst2.simulate(1000, 300)
    # analysis2 = Analyser(syst2)
    # energy = analysis2.get_energy(typ='lj')
    # analysis2.plot_energy(typ='lj', hline={"yval": 0, "color": "r", "style": "--"})
    # analysis2.plot_distribution(typ='energy')
    # analysis2.play_frames()
    # lj_potentials = []
    # for each in syst2.states():
    #
    #     lj_potentials.append(each.potential_lj())
    #
    # # print(syst.state.positions)
    # # print()
    #
    # syst.optimize(500)
    syst.simulate(2000, 300)

    # # print(syst.state.energy_lj)
    analysis = Analyser(syst)
    # energy = analysis.get_energy(typ='lj')
    # analysis.plot_energy(typ='lj', hline={"yval": 0, "color": "r", "style": "--"})
    # analysis.plot_distribution(typ='energy')
    rdf = analysis.plot_distribution(typ='rdf', bins=500)
    # lj_gro = np.load(r"D:\Uni\Master\CompSci\lj_md\lj_300.npy")
    #print(np.equal(energy, lj_gro))
    # plt.figure()
    plt.plot(rdf[:,0], rdf[:,1])
    # plt.plot(energy)
    plt.show()
    print("finished")
    # plt.close()
