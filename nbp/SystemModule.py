class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""
    def __init__(self, info, states):
        self.__systemInfo = info
        self.__systemStates = states

    def update_state(self, new_state):
        """Appends the new state to the systemStates list"""
        self.__systemStates.append(new_state)

    def info(self):
        """Gives the static information about the system"""
        return self.__systemInfo

    def state(self):
        """Gives the current dynamic information about the system"""
        pass

    def states(self):
        """Gives all the dynamic information about the system"""
        return self.__systemStates


class SystemInfo:
    """This class represents all the static information of the system

    __box_dim: an array 1, 2 or 3 dimensional, each cell is a dimension of the box containing the system [W, L, H]
    __particle_info:
    __sigma:
    __cutoff_radius: the radius chosen to do the cutoff
    """
    def __init__(self, box, particles, sigma):
        self.__box_dim = box
        self.__particle_info = particles
        self.__sigma = sigma
        self.__cutoff_radius = sigma * 2.5  # sigma * 2.5 is a standard approximation

    def box_dim(self):
        """Gives the box dimensions"""
        return self.__box_dim

    def particle_info(self):
        """Gives the static information about the particles"""
        return self.__particle_info

    def cutoff(self):
        """Returns the value chosen for the cutoff radius"""
        return self.__cutoff_radius


class SystemState:
    """Contains all the dynamic information about the system

    __positions: the position of the particles
    __electrostatics: the forces, the energies and the potentials of the particles
    __neighbours: the current status of the neighbours
    """
    def __init__(self, positions, electrostatics):
        self.__positions = positions
        self.__electrostatics = electrostatics
        self.__neighbours = None  # init the neighbours - don't know yet how

    def positions(self):
        """Returns the current particle positions
        SystemState.positions.shape = (num_particles, num_dimensions)"""
        return self.__positions

    def neighbours(self):
        """Returns the current neighbours list"""
        return self.__neighbours


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
