import numpy as np
import math


class Neighbours:

    def __init__(self, system_info, system_state):
        """
        Instantiates a new Object of class Neighbours
        :system: Instance of class System
        """
        self.SystemInfo = system_info
        self.SystemState = system_state
        self.__subcells_inrow = 1
        self.__subcell_length = self.__create_subcells()
        self.__neighbour_list = self.__create_neighbours()

        pass

    def __create_subcells(self):
        """
        Calculate the length of the subcells based on the skin radius.
        :return: Nothing?
        """
        # define skin radius by using sigma.
        # The 3 is just an option that mostly works (?)
        __skin_radius = self.SystemInfo.sigma() * 3
        __box_length = self.SystemInfo.char_length()

        # calculate subcell size.
        while __skin_radius < __box_length / self.__subcells_inrow:
            self.__subcells_inrow += 1

        subcell_length = __box_length / self.__subcells_inrow

        return subcell_length

    def __create_neighbours(self):
        """
        Creates the neighbours list by creating a head list which
        contains the starting index for the particles in a box.
        The neighbour list itself contains the index of the particles
        which belong to each box.
        :return: neighbour list
        """
        particle_number = self.SystemState.positions().shape[0]
        # create list (head) for starting index of subcell.
        # **3 because we have 3 dimensions for 2 it would be **2.
        self.__start_index = [0] * (self.__subcells_inrow ** 3)
        # create linked neighbour list
        self.__neighbour_list = [-1] * particle_number
        # get positions
        positions = self.SystemState.positions()

        for i in range(particle_number):

            self.__subcell_id = self.__find_subcell(positions[i])
            # print("position", i, ":", positions[i])
            try:
                self.__neighbour_list[i] = self.__start_index[self.__subcell_id]
                self.__start_index[self.__subcell_id] = i
            except IndexError:
                print("IndexError: Index out of range in start_index", self.__subcell_id)

        print("neighbour list:", self.__neighbour_list)

        return self.__neighbour_list

    def __find_subcell(self, position):
        """
        Private module of object neighbours. Finds the number of the
        subcell in which a particle is positioned.
        :param position: position of a particle
        :return: subcell_id
        """

        subcell_id_3d = [0, 0, 0]
        for axis in range(3):
            try:
                subcell_id_3d[axis] = math.floor(position[axis]
                                                 / self.__subcell_length)
                # print("subcell length: ", self.__subcell_length)
            except OverflowError:
                pass

        if (subcell_id_3d[0] < 0 or subcell_id_3d[1] < 0
                or subcell_id_3d[2] < 0):
            raise ValueError('subcell_id value is negative')
        else:
            # print("subcell_id_3d before:", subcell_id_3d)
            # print("subcells_inrow:", self.__subcells_inrow)
            m = self.__subcells_inrow
            self.__subcell_id = subcell_id_3d[0] \
                                + (subcell_id_3d[1] * m) \
                                + (subcell_id_3d[2] * (m ** 2))
            # print("subcell_id:", self.__subcell_id)

        return self.__subcell_id

    def update_neighbours(self):
        """
        :return:new neighbour list.
        """

        return self.__neighbour_list


    def get_neighbours(self, particle_position):
        """
        Issue: The particle counts itself so distances later should get rid of that!
        Calculates the particles which are neighbours of the particle which'
        position is given in 3d.
        :param particle_position: position of a single particle
        :return: neighbours
        """
        neighbours = []
        particle_subcell = self.__find_subcell(particle_position)
        m = self.__subcells_inrow

        # get subcell id for surrounding subcells
        # (including particle_subcell)
        # 8.1.18 no boundary subcells
        neighbour_subcells = [0] * 3 ** 3

        neighbour_subcells[13] = particle_subcell
        neighbour_subcells[12] = particle_subcell - 1
        neighbour_subcells[14] = particle_subcell + 1

        for i in range(1, 4):
            neighbour_subcells[14 + i] = particle_subcell + m + i - 1
            neighbour_subcells[12 - i] = particle_subcell - m - i + 3

        #for i in range(0, 8):
         #   neighbour_subcells[i] = neighbour_subcells[i + 9] - m**2
          #  neighbour_subcells[i + 18] = neighbour_subcells[i + 9] + m**2

        start_array = np.asarray(self.__start_index)
        print(start_array[0:26])
        print(neighbour_subcells)
        k = 1
        if k == 0:
          for i in np.nditer(neighbour_subcells):
            index = start_array[i]
            index = 1
            while index != 0:
                neighbours.append(index)
                self.__neighbour_list[index] = index

        return neighbours
