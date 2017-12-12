class System:
    """Wrapper for static SystemInfo and state dependent SystemState info."""

    __systemInfo
    __systemStates

    def __init__(self, info, states):

        self.__systemInfo = info
        self.__systemStates = states
        pass

    def update_state(self, newState):
        """Appends the new state to the systemStates list"""

    pass

    def give_info(self):
        """Gives the static information about the system"""
        return self.__systemInfo

    def give_state(self):
        """GIves the current dynamic information about the system"""

        pass

    def give_states(self):
        """Gives all the dynamic information about the system"""

        return self.__systemStates

    pass



class SystemInfo:

    __box_dim
    __particle_info


    def __init__(self, box, particles):

        self.__box_dim = box
        self.__particle_info = particles

        pass

    def give_box_dim(self):
        """Gives the box dimensions"""

        return self.__box_dim

    def give_particle_info(self):
        """Gives the static information about the particles"""

        return self.__particle_info


class SystemState:

    __positions
    __electrostatics
    __neighbours


    def __init__(self, positions, electrostatics):

        self.__positions = positions
        self.__electrostatics = electrostatics

        #init the neighbours

        pass

    def give_pos(self):
        """Returns the current particle positions"""

        return self.__positions

    def give_neighbours(self):
        """Returns the current neighbours list"""

        return self.__neighbours


class Electrostatic:

    __forces
    __potentials
    __energies


    def __init__(self):
        #maybe it should already calculate something?
        pass

    @property
    def potential(self):
        pass

    @property
    def energy(self):
        pass

    @property
    def forces(self):
        pass
