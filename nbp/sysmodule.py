import numpy as np
import scipy as sp
import nbp

from scipy.special import erfc


class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""

    def __init__(self, characteristic_length, sigma, epsilon_lj, particle_charges, positions,
                 lj=True, ewald=True, use_neighbours=False):
        particle_charges = np.asarray(particle_charges)
        sigma = np.asarray(sigma)
        epsilon_lj = np.asarray(epsilon_lj)

        if not (particle_charges.shape == sigma.shape):
            raise ValueError('Shapes do not agree: particle_charges, sigma.')

        if not (sigma.shape == epsilon_lj.shape):
            raise ValueError('Shapes do not agree: sigma and epsilon_lj.')

        positions = np.asarray(positions)

        if not (positions.shape[0] == epsilon_lj.shape[0]):
            raise ValueError('Shape[0]s do not agree: positions and epsilon_lj.')

        self._systemInfo = SystemInfo(characteristic_length, sigma, epsilon_lj, particle_charges, self,
                                      lj=lj, ewald=ewald, use_neighbours=use_neighbours)
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

    def __init__(self, characteristic_length, sigma, epsilon_lj, particle_charges, system,
                 lj=None, ewald=None, use_neighbours=None):
        self._sigma = np.asarray(sigma)
        self._worse_sigma = np.max(sigma)
        self._sigma_eff = (np.reshape(self._sigma[None, :], -1) + np.reshape(self._sigma, -1)[:, None])/2
        self._cutoff_radius = self._worse_sigma * 3  # 2.5 is standard, 3 is in neighbour list

        self._epsilon_lj = np.asarray(epsilon_lj)
        self._epsilon_lj_eff = np.sqrt(np.reshape(self._epsilon_lj, -1)[None, :]**2 +
                                       np.reshape(self._epsilon_lj, -1)[:, None]**2)
        self._epsilon0 = 1

        self._particle_charges = np.asarray(particle_charges)
        self._char_length = np.ceil(characteristic_length/self._cutoff_radius) * self._cutoff_radius
        self._system = system

        # booleans
        self._lj = lj
        self._ewald = ewald
        self._use_neighbours = use_neighbours

        if not isinstance(self._lj, bool):
            raise TypeError('_lj must be True (on) or False (off)')
        if not isinstance(self._ewald, bool):
            raise TypeError('_ewald must be True (on) or False (off)')
        if not isinstance(self._use_neighbours, bool):
            raise TypeError('_use_neighbours must be True (on) or False (off)')

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

    def sigma_eff(self):
        return self._sigma_eff

    def sigma(self):
        return self._sigma

    def worse_sigma(self):
        """Returns the maximum value of all the particles' couples' sigmas"""
        return self._worse_sigma

    def epsilon0(self):
        return self._epsilon0

    def epsilon_lj(self):
        return self._epsilon_lj

    def epsilon_lj_eff(self):
        return self._epsilon_lj_eff

    def particle_charges(self):
        return self._particle_charges

    def use_neighbours(self):
        return self._use_neighbours

    def lj(self):
        return self._lj

    def ewald(self):
        return self._ewald

    def num_particles(self):
        return self.particle_charges().shape[0]


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

        self._distance_vectors = None
        self._distances = None

        self._potential_lj = None
        self._energy_lj = None
        self._forces_lj = None

        self._potential_ewald = None
        self._energy_ewald = None
        self._forces_ewald = None

        self._potential = None
        self._energy = None
        self._forces = None

    def system(self):
        return self._system

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self._positions

    def distance_vectors(self):
        if self._distance_vectors is None:
            unwrapped = self.positions()[None, :, :] - self.positions()[:, None, :]
            wrapped = np.apply_along_axis(lambda x: nbp.periodic_wrap_corner(x, self.system().info().char_length()),
                                          -1, unwrapped)
            self._distance_vectors = wrapped
        return self._distance_vectors

    def distances(self):
        if self._distances is None:
            self._distances = np.linalg.norm(self.distance_vectors(), axis=-1)
        return self._distances

    def neighbours(self):
        if self._neighbours is None:
            self._neighbours = nbp.Neighbours(self.system().info(), self.system().state(), self.system(),
                                              verbose=self._verbose)
        return self._neighbours

    @staticmethod
    def calc_potential_lj(distance, epsilon_lj, sigma):
        """Calculates the potential between a couple of particles with a certain distance and a set sigma"""
        if sigma < 0:
            raise AttributeError('Sigma can\'t be smaller than zero')

        q = (sigma / distance)**6

        return 4.0 * epsilon_lj * (q * (q - 1))

    def potential_lj(self):
        """Calculates the Lennard-Jones potential between each couple of particles"""
        if self._potential_lj is None:
            if self.system().info().use_neighbours():
                self._potential = 0
                particle_number = self._positions.size
                for i in range(particle_number):
                    neighbour = self.neighbours().get_neighbours(self._positions[i])
                    for j in range(i + 1, particle_number):
                        sigma = self.system().info().sigma_eff()[i][neighbour.nb_pos[j]]
                        epsilon_lj = self.system().info().epsilon_lj_eff()[i][neighbour.nb_pos[j]]
                        distance = neighbour.nb_dist[j]
                        try:
                            pot_lj = self.calc_potential_lj(distance, epsilon_lj, sigma)
                            self._potential += pot_lj
                        except AttributeError:
                            print("Either sigma (={}) or the distance (={}) "
                                  "were wrongly calculated for the couple [{}][{}]".format(sigma, distance, i, j))
            else:
                out_shape = (self.system().info().num_particles(), self.system().info().num_particles())
                self._potential_lj = np.zeros(out_shape)
                for i in np.ndindex(out_shape):
                    self._potential_lj[i] = self.calc_potential_lj(self.distances()[i],
                                                                   self.system().info().epsilon_lj_eff()[i],
                                                                   self.system().info().sigma_eff()[i])
        return self._potential_lj

    def energy_lj(self):
        if self._energy_lj is None:
            if self.system().info().use_neighbours():
                pass
            else:
                self._energy_lj = np.sum(np.triu(self.potential_lj(), k=1))
        return self._energy_lj

    def forces_lj(self):
        if self._forces_lj is None:
            if self.system().info().use_neighbours():
                pass
            else:
                self._forces_lj = 0
        return self._forces_lj

    def potential_ewald(self):
        if self._potential_ewald is None:
            if self.system().info().use_neighbours():
                pass
            else:
                self._potential_ewald = 0
        return self._potential_ewald

    def energy_ewald(self):
        # Switch on columb versus lj
        # take the eqns from long range ewald, sub structure factors, use eulor/symm ->
        # couple interaction between two particles via ewald -> yeilds forces. (complex square of the structure factor)
        if self._energy_ewald is None:
            if self.system().info().use_neighbours():
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

                self._energy_ewald = energy_short + energy_long - energy_self
            else:
                pass
        return self._energy_ewald

    def forces_ewald(self):
        if self._forces_ewald is None:
            if self.system().info().use_neighbours():
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

                self._forces_ewald = forces_abs
            else:
                pass
        return self._forces_ewald

    def _check_lj_ewald(self, lj=None, ewald=None):
        if lj is None:
            lj = self._system.info().lj()
        if ewald is None:
            ewald = self._system.info().ewald()

        if not (isinstance(lj, bool) or isinstance(ewald, bool)):
            raise TypeError('LJ or Ewald were not selected properly in system initialization.')

        return lj, ewald

    def potential(self):
        lj, ewald = self._check_lj_ewald(lj=self.system().info().lj(),
                                         ewald=self.system().info().ewald())

        if self._potential is None:
            self._potential = np.zeros(self.system().info().num_particles(), 1)
            if lj:
                self._potential += self.potential_lj()
            if ewald:
                self._potential += self.potential_ewald()
        return self._potential

    def energy(self):
        lj, ewald = self._check_lj_ewald(lj=self.system().info().lj(),
                                         ewald=self.system().info().ewald())

        if self._energy is None:
            self._energy = 0
            if lj:
                self._energy += self.energy_lj()
            if ewald:
                self._energy += self.energy_ewald()
        return self._energy

    def forces(self):
        lj, ewald = self._check_lj_ewald(lj=self.system().info().lj(),
                                         ewald=self.system().info().ewald())

        if self._forces is None:
            self._forces = np.zeros(self.system().info().num_particles(), 3)
            if lj:
                self._forces += self.energy_lj()
            if ewald:
                self._forces += self.energy_ewald()
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
