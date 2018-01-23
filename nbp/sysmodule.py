import numpy as np
import scipy as sp
import nbp

from scipy.special import erfc


class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""

    def __init__(self, characteristic_length, sigma, particle_charges, positions):
        self._systemInfo = SystemInfo(characteristic_length, sigma, particle_charges, self)
        self._systemStates = [SystemState(positions, self)]
        self._MCMC = nbp.MCMC(self)

    def update_state(self, new_state):
        """Appends the new state to the systemStates list

        :param new_state: ????"""
        if isinstance(new_state, SystemState):
            self._systemStates.append(new_state)
        elif isinstance(new_state, list) and isinstance(new_state[0], SystemState):
            self._systemStates.extend(new_state)
        else:
            raise TypeError('SystemState could not be updated. Item is not type(SystemState) or a list of SystemState')

    def info(self):
        """Gives the static information about the system"""
        return self._systemInfo

    def state(self):
        """Gives the current dynamic information about the system"""
        return self._systemStates[-1]

    def states(self):
        """Gives all the dynamic information about the system"""
        return self._systemStates

    def optimize(self, max_steps, cov=None, d_energy_tol=1e-6, no_progress_break=10, num_particles=0.25):
        """Optimize the system to a lower energy level."""
        return self._MCMC.optimize(max_steps, cov=cov, d_energy_tol=d_energy_tol, no_progress_break=no_progress_break,
                                   num_particles=num_particles)

    def simulate(self, steps, temperature):
        """Simulate the system at a given temperature"""
        return self._MCMC.simulate(steps=steps, temperature=temperature)


class SystemInfo:
    """This class represents all the static information of the system

    characteristic_length = L in the notes, rounded up to the nearest n * cutoff_radius
    sigma: distance at which the inter-particle potential is zero
    worse_sigma: the biggest of all the sigmas
    cutoff_radius: the radius chosen to do the cutoff
    epsilon0: physical constant
    particle_charges: Arranged like position: (row, columns) == (particle_num, charge_value)
    """

    def __init__(self, characteristic_length, sigma, particle_charges, system):
        self._sigma = sigma
        self._worse_sigma = max(sigma)
        self._cutoff_radius = self._worse_sigma * 3 # 2.5 is standard, 3 is in neighbour list
        self._epsilon0 = 1
        self._particle_charges = np.asarray(particle_charges)
        self._char_length = np.ceil(characteristic_length/self._cutoff_radius) * self._cutoff_radius
        self._system = system

        if not self._cutoff_radius <= self._char_length/2:
            raise ValueError('The cutoff radius must be smaller than characteristic length divided by 2.')

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

    def worse_sigma(self):
        """Returns the maximum value of all the particles' couples' sigmas"""
        return self._worse_sigma


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

    def __init__(self, positions, system, verbose=False):
        self._verbose = verbose
        self._positions = np.asarray(positions)
        self._system = system
        self._neighbours = None
        self._potential = None
        self._energy = None
        # self._energy_lj = None
        self._forces = None

    def system(self):
        return self._system

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self._positions

    def neighbours(self):
        if self._neighbours is None:
            self._neighbours = nbp.Neighbours(self._system.info(), self._system.state(), self.system(),
                                              verbose=self._verbose)
        return self._neighbours

    def _potential_lj(self, epsilon, distance, sigma):
        """Calculates the potential between a couple of particles with a certain distance and a set sigma"""
        if sigma < 0:
            raise AttributeError('Sigma can\'t be smaller than zero')
        elif distance <= 0:
            raise AttributeError('The distance can\'t be smaller than or equal zero')

        q = (sigma / distance)**6

        return 4.0 * epsilon * (q * (q - 1))

    def potential(self, lj=True):
        """Calculates the Lennard-Jones potential between each couple of particles

            lj = a boolean variable that serves as a switch between Lennard Jones potential or DON'T KNOW YET THE OTHER
            :return the total potential of the system"""
        if self._potential is None:
            if lj:
                self._potential = 0
                particle_number = self._positions.size
                for i in range(particle_number):
                    neighbour = self.neighbours().get_neighbours(self._positions[i])
                    for j in range(i + 1, particle_number):
                        j_neighbour = neighbour.nb_pos[j]
                        sigma = self._system.sigma()[i][j_neighbour]
                        epsilon = self._system.epsilon()[i][j_neighbour]
                        distance = neighbour.nb_dist[j]
                        try:
                            pot_lj = self._potential_lj(epsilon, distance, sigma)
                            self._potential += pot_lj
                        except AttributeError:
                            print("Either sigma (={}) or the distance (={}) "
                                  "were wrongly calculated for the couple [{}][{}]".format(sigma, distance, i, j))
            else:
                """SPACE FOR OTHER POTENTIAL"""
        return self._potential

    @property
    def nrg(self, typ='total'):
        if typ == 'total':
            return self._energy() + self._potential()
        elif typ == 'lj':
            return self._potential()
        elif typ == 'coulomb':
            return self._energy()
        else:
            raise ValueError('The \"{}\" energy type is not understood. select one of the (total, lj, coulomb)'.format(typ))

    # # Ben's
    # def energy_lj(self):
    #     if self._energy_lj is None:
    #         epsilon = 1
    #         sigma = self.system().info().sigma()
    #         pos = self.positions()
    #         nb = self.neighbours()
    #
    #         energy = 0
    #         for particle in range(pos.shape[0]):
    #             rs = nb.get_neighbours(pos[particle]).nb_dist
    #             energy += np.sum(4*epsilon*((sigma/rs)**12 - (sigma/rs)**6))
    #
    #         self._energy_lj = energy
    #     return self._energy_lj

    def energy(self):
        # Switch on columb versus lj

        # take the eqns from long range ewald, sub structure factors, use eulor/symm ->
        # couple interaction between two particles via ewald -> yeilds forces. (complex square of the structure factor)
        if self._energy is None:
            V = self.system().info().volume()
            epsilon0 = self.system().info().epsilon0()
            charges = self.system().info().particle_charges()
            sigma = self.system().info().sigma()
            L = self.system().info().char_length()
            pos = self.positions()
            nb = self.neighbours()

            k_vectors = []
            reci_cutoff = 50  # Maybe put into system?
            for x in range(reci_cutoff):
                for y in range(-reci_cutoff, reci_cutoff, 1):
                    for z in range(-reci_cutoff, reci_cutoff, 1):
                        test = x+y+z
                        if test != 0:
                            k = [x, y, z]
                            k = [i * (2 * np.pi / L) for i in k]
                            k_vectors.append(k)

            # making sum for short energy
            shortsum = 0
            for i in range(len(pos)):
                neighbour = nb.get_neighbours(pos[i])
                for j in range(len(neighbour.nb_pos)):
                    if i != j:
                        distance = neighbour.nb_dist[j]
                        qi = charges[i]
                        qj = charges[neighbour.nb_pos[j]]
                        shortsum += (qi * qj) / (distance) * sp.special.erfc(
                            (np.linalg.norm(distance)) / (np.sqrt(2) * sigma))

            # making sum for long energy
            longsum = 0
            structure_factor = 0
            for x in range(len(k_vectors)):
                k = k_vectors[x]
                k_length = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
                for i in range(len(pos)):
                    q = charges[i]
                    r = pos[i]
                    structure_factor += 2 * q * np.cos(np.dot(k, r))
                    # its *2 because we calc only half the k vectors (symmetry)
                longsum += abs(structure_factor) ** 2 * np.exp(-sigma ** 2 * k_length ** 2 / 2) / k_length ** 2

            energy_short = 1 / (8 * np.pi * epsilon0) * shortsum
            energy_long = 1 / (V * epsilon0) * longsum
            energy_self = (2 * epsilon0 * sigma * (2 * np.pi) ** (3 / 2)) ** (-1) * np.sum(charges ** 2)

            self._energy = energy_short + energy_long - energy_self
        return self._energy

    def forces(self):
        if self._forces is None:
            pos = self.positions()
            charges = self.system().info().particle_charges()
            sigma = self.system().info().sigma()
            epsilon0 = self.system().info().epsilon0()
            nb = self.neighbours()
            L = self.system().info().char_length()
            forces_abs = []
            forces_near = []
            forces_far = []

            k_vectors = []
            reci_cutoff = 50  # Maybe put into system?
            for x in range(reci_cutoff):
                for y in range(-reci_cutoff, reci_cutoff, 1):
                    for z in range(-reci_cutoff, reci_cutoff, 1):
                        test = x+y+z
                        if test != 0:
                            k = [x, y, z]
                            k = [i * (2 * np.pi / L) for i in k]
                            k_vectors.append(k)

            # forces resulting from short energy
            for i in range(len(pos)):
                neighbour = nb.get_neighbours(pos[i])
                for j in range(len(neighbour.nb_pos)):
                    if i != j:
                        distance = neighbour.nb_dist[j]
                        qi = charges[i]
                        qj = charges[neighbour.nb_pos[j]]
                        energy = 1 / (8 * np.pi * epsilon0) * \
                                 (qi * qj)/distance * sp.special.erfc((np.linalg.norm(distance))/(np.sqrt(2) * sigma))
                        force = -np.diff(energy)/distance       # not sure about that one...
                        forces_near.append(force)

            # forces resulting from long energy
            structure_factor = 0
            for x in range(len(k_vectors)):
                k = k_vectors[x]
                k_length = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
                for i in range(len(pos)):
                    q = charges[i]
                    r = pos[i]
                    structure_factor += 2 * q * np.cos(np.dot(k, r))
                    # its *2 because we calc only half the k vectors (symmetry)
                    energy = abs(structure_factor) ** 2 * np.exp(-sigma ** 2 * k_length ** 2 / 2) / k_length ** 2
                    force = -np.diff(energy)/r                  # not sure about that one... again
                    forces_far.append(force)

            for i in range(len(forces_near)):
                forces_abs.append(forces_near[i]+forces_far[i])

            self._forces = forces_abs
        return self._forces


class FrameAnalysis:
    def __init__(self, system, positions_per_frame, energies_per_frame):
        self._system = system
        self._energies = energies_per_frame
        for i in positions_per_frame:
            self._system.update_state(i, self._system)

    def calc_energies(self):
        for i, v in enumerate(self._system.states()):
            calc_e = v.energy()
            given_e = self._energies
            diff = np.abs(calc_e - given_e)
            print("Frame {0} Calc_Energy {1} Given_Energy {2} Diff {3}".format(i, calc_e, given_e, diff))

# distance vector
# r in real^(N, D)
# np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
