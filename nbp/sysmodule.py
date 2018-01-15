import numpy as np
import scipy as sp
from scipy.special import erfc
from nbp import neighbours as nei
from scipy import exp, pi
from numpy import dot, sqrt

class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""
    def __init__(self, characteristic_length, sigma, particle_charges, positions):
        self.__systemInfo = SystemInfo(characteristic_length, sigma, particle_charges, self)
        self.__systemStates = [SystemState(positions, self)]

    def update_state(self, new_state):
        """Appends the new state to the systemStates list"""
        self.__systemStates.append(new_state)

    def info(self):
        """Gives the static information about the system"""
        return self.__systemInfo

    def state(self):
        """Gives the current dynamic information about the system"""
        return self.__systemStates[-1]

    def states(self):
        """Gives all the dynamic information about the system"""
        return self.__systemStates


class SystemInfo:
    """This class represents all the static information of the system

    characteristic_length = L in the notes
    sigma: Constant related to lennard jones
    cutoff_radius: the radius chosen to do the cutoff
    epsilon0: physical constant
    particle_charges: Arranged like position: (row, columns) == (particle_num, charge_value)
    """
    def __init__(self, characteristic_length, sigma, particle_charges, system, periods):
        self.__sigma = sigma
        self.__cutoff_radius = sigma * 2.5  # sigma * 2.5 is a standard approximation
        self.__epsilon0 = 1
        self.__particle_charges = particle_charges
        self.__char_length = characteristic_length
        self.__system = system
        self.__periods = periods

    def system(self):
        return self.__system

    def char_length(self):
        """Return the characteristic length aka L"""
        return self.__char_length

    def box_dim(self):
        """Gives the box dimensions.
        box_dim: a list, 3 dimensional, each cell is a dimension of the box containing the system [W, L, H]"""
        return [self.__char_length, self.__char_length, self.__char_length]

    def volume(self):
        """Returns the volume of the cell."""
        return self.__char_length**3

    def cutoff(self):
        """Returns the value chosen for the cutoff radius"""
        return self.__cutoff_radius

    def sigma(self):
        return self.__sigma

    def epsilon0(self):
        return self.__epsilon0

    def particle_charges(self):
        return self.__particle_charges

    def periods(self):
        return self.__periods


class SystemState:
    """Contains all the dynamic information about the system

    positions: the position of the particles (row, columns) == (particle_num, num_dimensions)
    electrostatics: the forces, the energies and the potentials of the particles
    neighbours: the current status of the neighbours
    """
    def __init__(self, positions, system):
        self.__positions = positions
        self.__neighbours = nei.Neighbours()
        self.__system = system

    def system(self):
        return self.__system

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self.__positions

    def neighbours(self):
        neigh = self.__neighbours.get_neighbours(self.positions())
        return neigh

    def potential(self):
        return None

    def energy(self):
        V = self.system().info().volume()
        epsilon0 = self.system().info().epsilon0()
        charges = self.system().info().particle_charges()
        sigma = self.system().info().sigma()
        L = self.system().info().char_length()
        pos = self.positions()
        n = self.system().info().periods()
        nb = self.neighbours()

        # start with one box
        # charge_matrix = np.outer(charges, charges.T)
        # np.fill_diagonal(charge_matrix, 0)
        # dist_matrix = np.fromfunction(lambda i, j: np.linalg.norm(pos[i] - pos[j]), (pos.shape[1], pos.shape[1]), dtype=int)
        # same_box = np.sum(charge_matrix/dist_matrix * sp.special.erf(dist_matrix/(np.sqrt(2)*sigma)))
        # now all other boxes
        # charge_matrix = np.outer(charges, charges.T)
        # other_boxes = None  # TODO

        # making sum for short energy
        shortsum = 0
        for i in range(len(nb.nb_pos)):
            for j in range(len(nb.nb_pos)):
                if i != j:
                    ri = pos[nb.nb_pos[i]]
                    rj = pos[nb.nb_pos[j]]
                    qi = charges[nb.nb_pos[i]]
                    qj = charges[nb.nb_pos[j]]
                    shortsum += (qi*qj) / (ri-rj + n*L) * sp.special.erfc((np.linalg.norm(ri-rj) + n*L)/(np.sqrt(2)*sigma))

        # making sum for long energy
        longsum = 0
        structure_factor = 0
        reci_cutoff = 100               # Maybe put into system?
        for x in range(reci_cutoff):
            for y in range(reci_cutoff):
                for z in range(reci_cutoff):
                    k = [x, y, z]
                    k = [x * (2*pi / L) for x in k]
                    k_length = sqrt(k[0]**2 + k[1]**2 + k[2]**2)
                    for i in range(len(pos)):                      # ToDo In range of NOT neighbour
                        q = charges[i]
                        r = pos[i]
                        structure_factor += q * exp(1j * dot(k, r))
                    longsum += abs(structure_factor)**2 * exp(-sigma**2 * k_length**2 / 2) / k_length**2

        energy_short = 1/(8*np.pi*epsilon0)*shortsum

        energy_long = 1/(V*epsilon0)*longsum

        energy_self = (2*epsilon0*sigma*(2*np.pi)**(3/2))**(-1)*np.sum(charges**2)
        return energy_short + energy_long - energy_self

    def forces(self):
        return None


class Error(Exception):
    """Base class for exceptions in this module"""
    pass
