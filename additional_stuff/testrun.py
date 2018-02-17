import numpy as np
# from nbp.misc import Analyser
from nbp.markov import *
from nbp.neighbours import *
from nbp.distance import *
from scipy.spatial import distance_matrix


class System:
    def __init__(self, parameters, positions=None, **kwargs):
        self._ipar = parameters
        self._systemInfo = SystemInfo(self)
        self._ipos = positions or self.generate_positions(self._ipar['num_particles'],
                                                          self._ipar['box'],
                                                          self._ipar['positions']
                                                          )
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

    @property
    def info(self):
        return self._systemInfo

    @property
    def state(self):
        """Gives the current dynamic information about the system"""
        return self._systemStates[-1]

    @property
    def states(self):
        """Gives all the dynamic information about the system"""
        return self._systemStates

    def optimize(self, *args, **kwargs):
        """Optimize the system to a lower energy level."""
        return self._MCMC.optimize(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """Simulate the system at a given temperature"""
        return self._MCMC.simulate(*args, **kwargs)


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

    @property
    def system(self):
        return self._system

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

    @property
    def system(self):
        return self._system

    def wrap_distances(self, distances):
        pass

    def wrap_positions(self, positions):
        R = [positions[:, 0], positions[:, 1], positions[:, 2]]
        dR = [self.system.info.box[0], self.system.info.box[1], self.system.info.box[2]]
        for i, r in enumerate(R):
            r[np.nonzero(r < dR[i])] = r[np.nonzero(r < dR[i])] + dR[i]
            r[np.nonzero(r >= dR[i])] = r[np.nonzero(r >= dR[i])] - dR[i]
            positions[:, i] = r
        return positions

    @property
    def pbc(self):
        return self._system.info.pbc

    @property
    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        if self.pbc:
            self._positions = self.wrap_positions(self._positions)
        return self._positions

    @property
    def distance(self):
        if self._distance is None:
            self._distance = np.triu(distance_matrix(self.positions, self.positions))
        return self._distance

    @staticmethod
    def lennard_jones(sigma, epsilon, distance):
        q = (sigma/distance)**6
        return 4*epsilon*q*(q-1)

    @property
    def energy_lj(self):
        if not self._energy_lj:
            self._energy_lj = 0
            for i in range(len(self.positions-1)):
                self._energy_lj += self.lennard_jones(
                    self.system.info.sigma_lj,
                    self.system.info.epsilon_lj,
                    self.distance[i, i+1]
                )
        return self._energy_lj


if __name__ == '__main__':


    parameters = {
        'num_particles': 4,     # number of particles
        'types': None,          # types of particles ('Ar', 'Na+') / ('Na+', 'Cl-')
        'epsilon_lj': 0.9363,   # in kJ/mol
        'sigma_lj': 0.340,      # in nanometers
        'charges': None,        # charges tuple ('Ar', 0) / ('Na+', 0.345) / ('Cl-', 0.345)
        'box': [5, 5, 5],       # box dimensions
        'positions': 'random',  # method for positions generation (random or grid)
        'pbc': True
    }
    sys = System(parameters)
    sys.optimize(500)
    sys.simulate(100, 300)
