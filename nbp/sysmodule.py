import numpy as np
import scipy as sp
import nbp

from scipy.special import erfc


class System:
    """This class represents the system itself: the box and its content.
    It's a wrapper containing SystemInfo and all the SystemStates, mostly. Unless stated, all the attributes in
    the form of an array or a matrix follow the following convention:
        information[i] contains information about the particle i
        pairInformation[i][j] contains information about the relation between particle i and j (e.g. the distance).
    Attributes:
        particle_charges:   ndarray
                            the charges of the particle.
        sigma:  ndarray
                the value of a particle's sigma.
        epsilon_lj: ndarray
                    the value of a single particle's epsilon.
        _systemInfo:    SystemInfo
                        the object containing all the static information about the system.
        _systemStates:  array
                        contains the state of the system at each time step.
                        (_systemStates[i] contains the state at step i)
        _MCMC:  MCMC
                the actor who simulate or optimize the system.
    """

    def __init__(self, characteristic_length, sigma, epsilon_lj, particle_charges, positions, reci_cutoff,
                 lj=True, ewald=True, use_neighbours=False):
        """The initialize function.
        :param  characteristic_length:  float
                                       the characteristic length.
        :param  sigma:  ndarray
                        the value of a particle's sigma.
        :param  epsilon_lj: ndarray
                            the value of a signle particle's epsilon.
        :param  particle_charges:   ndarray
                                    the charges of the particles.
        :param  positions:  ndarray
                            the position of the particles.
        :param  lj: boolean, optional (default = True)
                    if true, the Lennard Jones potential is used in the energy calculations.
        :param  ewald:  boolean, optional (default = True)
                        if true, Ewald's summation is used in the energy calculations.
        :param  use_neighbours: boolean, optional (default = False)
                                if true, a cut off radius is applied during the calculations.
        """
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

        self._systemInfo = SystemInfo(characteristic_length, sigma, epsilon_lj, particle_charges, reci_cutoff, self,
                                      lj=lj, ewald=ewald, use_neighbours=use_neighbours)
        self._systemStates = [SystemState(positions, self)]
        self._MCMC = nbp.MCMC(self)

    def update_state(self, new_state):
        """Appends the new state to the systemStates list.
        :param  new_state:  SystemState
                        the state at step len(self._systemState) + 1.
        :return nothing
        """
        if isinstance(new_state, SystemState):
            self._systemStates.append(new_state)
        elif isinstance(new_state, list) and isinstance(new_state[0], SystemState):
            self._systemStates.extend(new_state)
        else:
            raise TypeError('SystemState could not be updated. Item is not type(SystemState) or a list of SystemState')

    def info(self):
        """Gives the static information about the system.
        :return
            SystemInfo
            the object containing all the static information about the system.
            """
        return self._systemInfo

    def state(self):
        """Gives the current dynamic information about the system.
        :return
            SystemState
            the object containing all the dynamic information about the system.
            """
        return self._systemStates[-1]

    def states(self):
        """Gives all the dynamic information past and present about the system.
        :return
            list
            a list containing all the states of the system, past and present.
        """
        return self._systemStates

    # def optimize(self, max_steps=500, cov=None, d_energy_tol=1e-6, no_progress_break=250, num_particles=0.25):
    def optimize(self, *args, **kwargs):
        """Optimize the system to a lower energy level.
        :arg
            max_steps:  int, optional (default = 500)
                        the maximum number of steps that the optimizer can make before stopping.
            cov:    I HAVE NO IDEA, optional (default = None)
                    KEEP HAVING NO IDEA
            d_energy_tol:   float , optional (default = 1e-6)
                            I HAVE NO IDEA
            no_progress_break: int, optional (default = 250)
                                I HAVE NO IDEA
            num_particles:  float, optional (default = 0.25)
                            I HAVE NO IDEA, I SUPPOSED IT'S THE NUMBER OF PARTICLES, BUT 0.25 DOESN'T MAKE SENSE
        :return
            SystemState
            the optimized state
        """
        return self._MCMC.optimize(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """Simulate the system at a given temperature
        :arg
            steps:  int
                    the number of steps to simulate
            temperature:    float
                            the temperature at which the system has to be simulated (in K)
        :return
            SystemState
            the state after steps number of steps"""
        return self._MCMC.simulate(*args, **kwargs)


class SystemInfo:
    """This class represents all the static information of the system.
    Unless stated, all the attributes in the form of an array or a matrix follow the following convention:
        information[i] contains information about the particle i
        pairInformation[i][j] contains information about the relation between particle i and j (e.g. the distance).
    Attributes:
        _char_length:   float
                        the length of side of the cubic box
        _particle_charges:  ndarray
                            the charges of the particle.
        _sigma:  ndarray
                the value of a particle's sigma.
        _worse_sigma:   float
                        the biggest sigma of all the present sigmas.
        _cutoff_radius: float
                        the radius of the cutoff.
        _epsilon_lj:    ndarray
                        the value of a single particle's epsilon.
        _epsilon_lj_eff:    ndarray
                            all the epsilon relative to each couple of particles.
        _epsilon0:  float
                    physical constant.
        _system:    System
                    the system containing this systemInfo.
        _lj:    Boolean
                if True, LJ potential is used in the energies calculations.
        _ewald: Boolean
                if True, Ewald's summation is used in the energies calculations.
        _use_neighbours:    Boolean
                            if True, the neighbourlist is implemented.
    """

    def __init__(self, characteristic_length, sigma, epsilon_lj, particle_charges, reci_cutoff, system,
                 lj=None, ewald=None, use_neighbours=None):
        """Initialization function.
        :param characteristic_length:  float
                                       the length of the cubic box's side.
        :param  sigma:  ndarray
                        the value of a particle's sigma.
        :param  epsilon_lj: ndarray
                            the value of a single particle's epsilon.
        :param  particle_charges:   ndarray
                                    the charges of the particle.
        :param  system: System
                        the system containing this systemState
        :param  _lj:    Boolean
                       if True, LJ potential is used in the energies calculations.
        :param  _ewald: Boolean
                        if True, Ewald's summation is used in the energies calculations.
        :param  _use_neighbours:    Boolean
                                    if True, the neighbourlist is implemented.
        """
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

        # k vectors
        self._k_vectors = []
        for x in range(reci_cutoff):
            for y in range(-reci_cutoff, reci_cutoff, 1):
                for z in range(-reci_cutoff, reci_cutoff, 1):
                    test = x + y + z
                    if test != 0:
                        k = [x, y, z]
                        k = [i * (2 * np.pi / self._char_length) for i in k]
                        self._k_vectors.append(k)

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
        """Returns the system containing self.
        :return System
                the system containing self."""
        return self._system

    def char_length(self):
        """Return the characteristic length of the box.
        :return float
                the characteristic length of the box."""
        return self._char_length

    def box_dim(self):
        """Returns the box dimensions.
        :return list
                the three dimension of the box as a list: each cell is a
                dimension of the box containing the system [W, L, H]"""
        return [self._char_length, self._char_length, self._char_length]

    def volume(self):
        """Returns the volume of the cell.
        :return float
                the volume of the box."""
        return self._char_length ** 3

    def cutoff(self):
        """Returns the value chosen for the cutoff radius.
        :return float
                the cutoff radius"""
        return self._cutoff_radius

    def sigma_eff(self):
        """Returns the matrix containing all the sigma relative to the particles couples
        :return ndarray
                an array of arrays, each cell [i][j] contains sigma_ij"""
        return self._sigma_eff

    def sigma(self):
        """Returns all the particles' sigmas
        :return ndarray
                an array containing each particle's sigma"""
        return self._sigma

    def worse_sigma(self):
        """Returns the maximum value of all the particles' couples' sigmas.
        :return float
                the biggest sigma."""
        return self._worse_sigma

    def epsilon0(self):
        """Returns the physical constant epsilon.
        :return float
                epsilon"""
        return self._epsilon0

    def k_vectors(self):
        return self._k_vectors

    def epsilon_lj(self):
        """Returns all the particles' epsilon.
        :return ndarray
                an array containing each particle's epsilon."""
        return self._epsilon_lj

    def epsilon_lj_eff(self):
        """Returns the matrix containing all the epsilons relative to the particles couples
        :return ndarray
                an array of arrays, each cell [i][j] contains epsilon_ij"""
        return self._epsilon_lj_eff

    def particle_charges(self):
        """Returns all the particles' charges.
        :return ndarray
                an array containing each particle's charge."""
        return self._particle_charges

    def use_neighbours(self):
        """Flag saying if the neighbours must be used.
        :return Boolean
                if True, the neighbourlist is being used."""
        return self._use_neighbours

    def lj(self):
        """Flag saying if LJ potential is used in the calculations.
        :return Boolean
                if True, LJ potential is used in the calculations."""
        return self._lj

    def ewald(self):
        """Flag saying if Ewald summation is used in the calculations.
        :return Boolean
                if True, Ewald's summation is used in the calculations."""
        return self._ewald

    def num_particles(self):
        """Returns the number of particles of the system.
        :return int
                the number of particles."""
        return self.particle_charges().shape[0]


class SystemState:
    """Contains all the dynamic information about the system.
    Unless stated, all the attributes in the form of an array or a matrix follow the following convention:
        information[i] contains information about the particle i
        pairInformation[i][j] contains information about the relation between particle i and j (e.g. the distance).
    Instance attributes:
        _verbose:   Boolean
                    I HAVE NO IDEA
        _system:    System
                    the system containing this systemState.
        _positions: ndarray
                    a n-dimensional array containing the 3D positions of each particle.
        _neighbours:    Neighbours
                        the neighbourhood.
        _distance:  ndarray
                    the distances between all the particles, as an array.
        _potential_lj:  ndarray
                        the LJ potential between all the particles.
        _energy_lj: float
                    the energy calculated using Lennard Jones.
        _forces_lj: NOIDEA
                    the forces calculated using Lennard Jones.
        _potential_ewald:   NOIDEA
                            the potential calculated using Ewald's summation.
        _forces_ewald:  NOIDEA
                        the forces calculated using Lennard Jones.
        _potential: float CREDO
                    the total potential.
        _energy:    float
                    the total system's energy.
        _forces:    NOIDEA
                    the forces in the whole system.
    """

    def __init__(self, positions, system, verbose=False):
        """
        :param positions:   ndarray
                            the positions of the particles.
        :param system:  System
                        the system containing self.
        :param verbose: Boolean
                        I HAVE NO IDEA.
        """
        self._verbose = verbose
        self._system = system
        self._positions = nbp.periodic_particles_stay_in_box(np.asarray(positions), self.system().info().char_length())
        self._neighbours = None
        self._distance = None

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
        """Returns the system containing self.
        :return System
                the system containing self."""
        return self._system

    def neighbours(self):
        """Creates or returns an instance of the class Neighbours.
        :return: Instance of neighbours"""
        if self._neighbours is None:
            self._neighbours = nbp.Neighbours(self.system().info(), self.system().state(), self.system(),
                                              verbose=self._verbose)
        return self._neighbours

    def distance(self):
        """Initialize class distance for later use.
        :return: instance of class Distance"""
        if self._distance is None:
            self._distance = nbp.Distance(self.system())
        return self._distance

    @staticmethod
    def calc_potential_lj(distance, epsilon_lj, sigma):
        """Calculates the potential between a couple of particles with a certain distance and a set sigma
        :param distance:    float
                            the distance between the two particles.
        :param epsilon_lj:  float
                            epsilon between the two particles.
        :param sigma:   float
                        sigma between the two particles.
        :return float
                the Lennard Jones potential between the two particles."""

        if distance != 0:
            q = (sigma / distance)**6
        else:
            q = 0                   # ToDo: Like this Ben?

        return 4.0 * epsilon_lj * (q * (q - 1))

    def positions(self):
        """Returns the current particle positions.
        :return ndarray
                the particles's positions."""
        self._positions = nbp.periodic_particles_stay_in_box(self._positions, self.system().info().char_length())

        return self._positions

    def potential_lj(self):
        """Calculates the Lennard-Jones potential between each couple of particles.
        :return ndarray
                an array of arrays containing the potential between each couple of particles."""
        if self._potential_lj is None:
            if self.system().info().use_neighbours():
                self._energy_lj = 0
                self._potential_lj = np.zeros((self.system().info().num_particles(),
                                               self.system().info().num_particles()))
                for i in range(self.system().info().num_particles()):
                    neighbours = self.neighbours().get_neighbours(i)
                    for j in range(len(neighbours.nb_ID)):
                        j_neighbor = neighbours.nb_ID[j]
                        if j_neighbor > i:
                            sigma = self.system().info().sigma_eff()[i][j_neighbor]
                            epsilon = self.system().info().epsilon_lj_eff()[i][j_neighbor]
                            distance = neighbours.nb_dist[j]
                            pot_lj = self.calc_potential_lj(distance, epsilon, sigma)
                            self._potential_lj[i][j_neighbor] = pot_lj
                            self._potential_lj[j_neighbor][i] = pot_lj
                            self._energy_lj += pot_lj
            else:
                out_shape = (self.system().info().num_particles(), self.system().info().num_particles())
                self._potential_lj = np.zeros(out_shape)
                for i in np.ndindex(out_shape):
                    self._potential_lj[i] = self.calc_potential_lj(self.distance().distances_wrapped()[i],
                                                                   self.system().info().epsilon_lj_eff()[i],
                                                                   self.system().info().sigma_eff()[i])
        return self._potential_lj

    def energy_lj(self):
        """Calculates the energy through Lennard Jones Potential.
        :return float
                the total energy through Lennard Jones Potential."""
        if self._energy_lj is None:
            self.potential_lj()
        return self._energy_lj

    def forces_lj(self):
        """Calculates the forces acting on every particle.
        :return NOIDEA
                the forces acting on every particle."""
        if self._forces_lj is None:
            if self.system().info().use_neighbours():
                pass
            else:
                self._forces_lj = 0
        return self._forces_lj

    def potential_ewald(self):
        """Calculates the potential through Ewald summation.
        :return float
                the potential calculated through Ewald summation."""
        if self._potential_ewald is None:
            if self.system().info().use_neighbours():
                pass
            else:
                self._potential_ewald = 0
        return self._potential_ewald

    def energy_ewald(self):
        """Calculates the energies using Ewald summation.
        :return float
                the total energy calculated through Ewald summation."""
        # Switch on columb versus lj
        # take the eqns from long range ewald, sub structure factors, use eulor/symm ->
        # couple interaction between two particles via ewald -> yeilds forces. (complex square of the structure factor)
        if self._energy_ewald is None:
            V = self.system().info().volume()
            epsilon0 = self.system().info().epsilon0()
            charges = self.system().info().particle_charges()
            sigma = self.system().info().sigma_eff()
            sigma_one = self.system().info().sigma()
            pos = self.positions()
            cutoff = self.system().info().cutoff()
            k_vectors = self.system().info().k_vectors()

            if self.system().info().use_neighbours():
                # making sum for short energy WITH neighbours
                nb = self.neighbours()
                shortsum = 0
                for i in range(len(pos)):
                    neighbour = nb.get_neighbours(i)
                    for j in range(len(neighbour.nb_ID)):
                        if i != j:
                            distance = neighbour.nb_dist[j]
                            qi = charges[i]
                            qj = charges[neighbour.nb_ID[j]]
                            shortsum += (qi * qj) / distance * sp.special.erfc(
                                (np.linalg.norm(distance)) / (np.sqrt(2) * sigma[i, j]))
            else:
                # making sum for short energy WITHOUT neighbours
                shortsum = 0
                for i in range(len(pos)):
                    for j in range(len(pos)):
                        if i != j:
                            vector = pos[i] - pos[j]
                            distance = np.sqrt(sum(x**2 for x in vector))
                            if distance < cutoff:
                                qi = charges[i]
                                qj = charges[j]
                                shortsum += (qi * qj) / distance * sp.special.erfc(
                                    distance / (np.sqrt(2) * sigma[i, j]))

            # making sum for long energy
            longsum = 0
            structure_factor = 0
            for x in range(len(k_vectors)):
                k = k_vectors[x]
                k_length = np.linalg.norm(k)
                for i in range(len(pos)):
                    q = charges[i]
                    r = pos[i]
                    structure_factor += 2 * q * np.cos(np.dot(k, r))
                    # its *2 because we calc only half the k vectors (symmetry)
                longsum += np.linalg.norm(structure_factor) ** 2 * np.exp((-np.mean(sigma_one) ** 2 * k_length ** 2) / 2) / k_length ** 2
                # ToDo What Sigma here?

            # making sum for self energy
            selfsum = 0
            for i in range(len(charges)):
                whyArray = sigma_one[i] * charges[i]**2
                selfsum += whyArray[0]

            energy_short = 1 / (8 * np.pi * epsilon0) * shortsum
            energy_long = 1 / (V * epsilon0) * longsum
            energy_self = (2 * epsilon0 * (2 * np.pi) ** (3 / 2)) ** (-1) * selfsum

            # print("short is ")
            # print(energy_short)
            # print("long is ")
            # print(energy_long)
            # print("self is ")
            # print(energy_self)

            self._energy_ewald = energy_short + energy_long - energy_self
        return self._energy_ewald

    def forces_ewald(self):
        if self._forces_ewald is None:
            pos = self.positions()
            charges = self.system().info().particle_charges()
            sigma = self.system().info().sigma()
            epsilon0 = self.system().info().epsilon0()
            nb = self.neighbours()
            L = self.system().info().char_length()
            V = self.system().info().volume()
            cutoff = self.system().info().cutoff()
            forces_abs = []
            forces_near = []
            forces_far = []

            k_vectors = []
            reci_cutoff = 10
            for x in range(reci_cutoff):
                for y in range(-reci_cutoff, reci_cutoff, 1):
                    for z in range(-reci_cutoff, reci_cutoff, 1):
                        test = x + y + z
                        if test != 0:
                            k = [x, y, z]
                            k = [i * (2 * np.pi / L) for i in k]
                            k_vectors.append(k)

            # forces resulting from short energy
            for i in range(len(pos)):
                if self.system().info().use_neighbours():
                    neighbour = nb.get_neighbours(i)
                    force_sum = 0
                    for j in range(len(neighbour.nb_ID)):
                        if i != j:
                            distance = neighbour.nb_dist[j]
                            qj = charges[neighbour.nb_ID[j]]
                            force_sum += qj * distance / distance**2 \
                                         * ( sp.special.erfc( np.linalg.norm(distance)/np.sqrt(2)*sigma[i, j])/ np.linalg.norm(distance)) \
                                        + np.sqrt(2/np.pi) * sigma[i, j]**(-1) * np.exp(- np.linalg.norm(distance)**2 / 2* sigma[i, j]**2)
                    force_sum = charges[i] / (8 * np.pi * epsilon0) * force_sum
                    forces_near.append(force_sum)
                else:
                    force_sum = 0
                    for j in range(len(charges)):
                        vector = pos[i] - pos[j]
                        distance = np.sqrt(sum(x ** 2 for x in vector))
                        if distance < cutoff:
                            qj = charges[j]
                            force_sum += qj * distance / distance ** 2 \
                                         * (sp.special.erfc(
                                np.linalg.norm(distance) / np.sqrt(2) * sigma[i, j]) / np.linalg.norm(distance)) \
                                         + np.sqrt(2 / np.pi) * sigma[i, j] ** (-1) * np.exp(
                                - np.linalg.norm(distance) ** 2 / 2 * sigma[i, j] ** 2)
                    force_sum = charges[i] / (8 * np.pi * epsilon0) * force_sum
                    forces_near.append(force_sum)

            # forces resulting from long energy
            structure_factor = 0
            for i in range(len(pos)):
                qi = charges[i]
                for x in range(len(k_vectors)):
                    k = k_vectors[x]
                    k_length = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
                    for j in range(len(pos)):
                        q = charges[j]
                        r = pos[j]
                        structure_factor += (np.exp(- sigma[i]**2 * k_length**2 / 2) / k_length**2 ) * 2 * q * np.cos(np.dot(k, r)) ** 2 * k
                force_sum = qi / (V * epsilon0)
                forces_far.append(force_sum)

            for i in range(len(forces_near)):
                forces_abs.append([forces_near[i], forces_far[i]])

            self._forces_ewald = forces_abs
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
            # print('energy:', type(self._energy))
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
