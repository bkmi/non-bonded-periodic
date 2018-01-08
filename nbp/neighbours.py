from sysmodule import System
import math


class Neighbours:

    def __init__(self, system):
        """
        Instantiates a new Object of class Neighbours
        :system: Instance of class System
        """
        self.System = system
        self.__neighbour_list = self.__create_neighbours()

    def __create_subcells(self):
        """
        Calculate the length of the subcells based on the skin radius.
        :return: Nothing?
        """
        # define skin radius by using sigma.
        # The 3 is just an option that mostly works (?)
        __skin_radius = self.System.info.sigma * 3
        __box_length = self.System.state.char_length

        self.__subcells_inrow = 1  # initialize

        # calculate subcell size.
        while __skin_radius < __box_length / self.__subcells_inrow:
            self.__subcells_inrow += 1

        self.__subcell_length = __box_length / self.__subcells_inrow

        pass

    def __create_neighbours(self):
        """
        Creates the neighbours list by creating a head list which
        contains the starting index for the particles in a box.
        The neighbour list itself contains the index of the particles
        which belong to each box.
        :return: neighbour list
        """
        particle_number = self.System.state.positions().shape[0]
        # create list (head) for starting index of subcell.
        # ^3 because we have 3 dimensions for 2 it would be ^2.
        start_index = [0] * (self.__subcells_inrow ^ 3)
        # create linked neighbour list
        self.__neighbour_list = []
        # get positions
        positions = self.System.state.positions()

        for i in range(0, particle_number - 1):

            subcell_id = self.__find_subcell(positions[i])

            self.__neighbour_list[i] = start_index[subcell_id]
            start_index[subcell_id] = i

        return self.__neighbour_list

    def __find_subcell(self, position):
        """
        Private module of object neighbours. Finds the number of the
        subcell in which a particle is positioned.
        :param position: position of a particle
        :return: subcell_id
        """
        subcell_id = [-1, -1, -1]
        for axis in range(0, 2):
            subcell_id[axis] = math.floor(position[axis]
                                          / self.__subcell_length)

        if (subcell_id[0] < 0 | subcell_id[1] < 0
                | subcell_id[2] < 0):
            print("Error: subcell_id value is negative!")
            # throw exception?
        else:
            m = self.__subcells_inrow
            subcell_id = subcell_id[0] \
                + (subcell_id[1] - 1) * m \
                + (subcell_id[2] - 1) * m ^ 2

        return subcell_id

    def update_neighbours(self):
        """
        :return:new neighbour list.
        """

        return self.__neighbour_list
