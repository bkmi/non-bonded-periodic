import numpy as np
import scipy as sp
import nbp

from scipy.special import erfc


class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""

    def __init__(self, characteristic_length, sigma, particle_charges, positions):
        self._systemInfo = SystemInfo(characteristic_length, sigma, particle_charges, self)
        init_state = SystemState(positions, self)
        self._systemStates = [init_state]

    def update_state(self, new_state):
        """Appends the new state to the systemStates list"""
        self._systemStates.append(new_state)

    def info(self):
        """Gives the static information about the system"""
        return self._systemInfo

    def state(self):
        """Gives the current dynamic information about the system"""
        return self._systemStates[-1]

    def states(self):
        """Gives all the dynamic information about the system"""
        return self._systemStates

    def optimize(self):
        """Optimize the system to a lower energy level."""
        pass

    def simulate(self):
        """Simulate the system at a given temperature"""
        pass


class SystemInfo:
    """This class represents all the static information of the system

    characteristic_length = L in the notes
    sigma: Constant related to lennard jones
    cutoff_radius: the radius chosen to do the cutoff
    epsilon0: physical constant
    particle_charges: Arranged like position: (row, columns) == (particle_num, charge_value)
    """

    def __init__(self, characteristic_length, sigma, particle_charges, system):
        self._sigma = sigma
        self._cutoff_radius = sigma * 2.5  # sigma * 2.5 is a standard approximation
        self._epsilon0 = 1
        self._particle_charges = particle_charges
        self._char_length = characteristic_length
        self._system = system

    def system(self):
        return self._system

    def char_length(self):
        """Return the characteristic length aka L"""
        return self._char_length

    def box_dim(self):
        """Gives the box dimensions.
        box_dim: a list, 3 dimensional, each cell is a dimension of the box containing the system [W, L, H]"""
        return [self._char_length, self._char_length, self._char_length]

    def volume(self):
        """Returns the volume of the cell."""
        return self._char_length ** 3

    def cutoff(self):
        """Returns the value chosen for the cutoff radius"""
        return self._cutoff_radius

    def sigma(self):
        return self._sigma

    def epsilon0(self):
        return self._epsilon0

    def particle_charges(self):
        return self._particle_charges


class SystemState:
    """Contains all the dynamic information about the system

    positions: the position of the particles (row, columns) == (particle_num, num_dimensions)
    electrostatics: the forces, the energies and the potentials of the particles
    neighbours: the current status of the neighbours
    """

    def __init__(self, positions, system):
        self._positions = positions
        self._system = system
        self._neighbours = None
        self._potential = None
        self._energy = None
        self._forces = None

    def system(self):
        return self._system

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self._positions

    def neighbours(self):
        if self._neighbours is None:
            self._neighbours = nbp.Neighbours(self._system.info(), self._system.state())
        return self._neighbours

    def potential(self):
        if self._potential is None:
            self._potential = 0
        return self._potential

    def energy(self):
        if self._energy is None:
            self.neighbours().set_neighbours(self.positions())

            V = self.system().info().volume()
            epsilon0 = self.system().info().epsilon0()
            charges = self.system().info().particle_charges()
            sigma = self.system().info().sigma()
            L = self.system().info().char_length()
            pos = self.positions()
            n = self.system().info().periods()
            nb = self.neighbours()

            # making sum for short energy
            shortsum = 0
            for i in range(len(nb.nb_pos)):
                for j in range(len(nb.nb_pos)):
                    if i != j:
                        ri = pos[nb.nb_pos[i]]
                        rj = pos[nb.nb_pos[j]]
                        qi = charges[nb.nb_pos[i]]
                        qj = charges[nb.nb_pos[j]]
                        shortsum += (qi * qj) / (ri - rj + n * L) * sp.special.erfc(
                            (np.linalg.norm(ri - rj) + n * L) / (np.sqrt(2) * sigma))

            # making sum for long energy
            longsum = 0
            structure_factor = 0
            reci_cutoff = 100  # Maybe put into system?
            for x in range(reci_cutoff):
                for y in range(reci_cutoff):
                    for z in range(reci_cutoff):
                        k = [x, y, z]
                        k = [x * (2 * np.pi / L) for x in k]
                        k_length = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
                        for i in range(len(pos)):  # ToDo In range of NOT neighbour
                            q = charges[i]
                            r = pos[i]
                            structure_factor += q * np.exp(1j * np.dot(k, r))
                        longsum += abs(structure_factor) ** 2 * np.exp(-sigma ** 2 * k_length ** 2 / 2) / k_length ** 2

            energy_short = 1 / (8 * np.pi * epsilon0) * shortsum

            energy_long = 1 / (V * epsilon0) * longsum

            energy_self = (2 * epsilon0 * sigma * (2 * np.pi) ** (3 / 2)) ** (-1) * np.sum(charges ** 2)

            self._energy = energy_short + energy_long - energy_self
        return self._energy

    def forces(self):
        if self._forces is None:
            self._forces = 0
        return self._forces
