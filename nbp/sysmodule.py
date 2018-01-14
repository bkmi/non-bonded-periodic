from .markov import MCMC


class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""
    def __init__(self, characteristic_length, sigma, particle_charges, positions):
        self.__systemInfo = SystemInfo(characteristic_length, sigma, particle_charges)
        self.__systemStates = [SystemState(positions)]
        self.__mcmc = MCMC(self)

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
    def __init__(self, characteristic_length, sigma, particle_charges):
        self.__sigma = sigma
        self.__cutoff_radius = sigma * 2.5  # sigma * 2.5 is a standard approximation
        self.__epsilon0 = 1
        self.__particle_charges = particle_charges
        self.__char_length = characteristic_length

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


class SystemState:
    """Contains all the dynamic information about the system

    positions: the position of the particles (row, columns) == (particle_num, num_dimensions)
    electrostatics: the forces, the energies and the potentials of the particles
    neighbours: the current status of the neighbours
    """
    def __init__(self, positions):
        self.__positions = positions
        self.__electrostatics = Electrostatic()
        self.__neighbours = None  # init the neighbours - don't know yet how

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self.__positions

    def neighbours(self):
        """Returns the current neighbours list"""
        return self.__neighbours

    def electrostatics(self):
        """Return the current state of the electrostatics"""
        return self.__electrostatics


class Electrostatic:
    """Represent the electrostatics information of the system

    __forces:
    __potentials:
    __energies:
    """
    def __init__(self):
        # think
        self.__potential = None
        self.__energy = None
        self.__forces = None
        pass

    @property
    def potential(self):
        return self.__potential

    @property
    def energy(self):
        return self.__energy

    @property
    def forces(self):
        return self.__forces


class Error(Exception):
    """Base class for exceptions in this module"""
    pass
