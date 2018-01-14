import numpy as np
import math
import collections

class Neighbours:

    def __init__(self, system_info, system_state):
        """
        Instantiates a new Object of class Neighbours
        :system: Instance of class System
        """
        self.SystemInfo = system_info
        self.SystemState = system_state
        self.__box_length = self.SystemInfo.char_length()
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

        # calculate subcell size.
        while __skin_radius < self.__box_length / self.__subcells_inrow:
            self.__subcells_inrow += 1
        print("Number of subcells:", self.__subcells_inrow)
        subcell_length = self.__box_length / self.__subcells_inrow

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

            subcell_id = self.__find_subcell(positions[i])
            # print("position", i, ":", positions[i])
            try:
                self.__neighbour_list[i] = self.__start_index[subcell_id]
                self.__start_index[subcell_id] = i
            except IndexError:
                print("IndexError: Index out of range in start_index", subcell_id)

        #print("neighbour list:", self.__neighbour_list)

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
            except OverflowError:
                pass

        if (subcell_id_3d[0] < 0 or subcell_id_3d[1] < 0
                or subcell_id_3d[2] < 0):
            raise ValueError('subcell_id value is negative')
        else:
            m = self.__subcells_inrow
            subcell_id = subcell_id_3d[0] \
                         + (subcell_id_3d[1] * m) \
                         + (subcell_id_3d[2] * (m ** 2))
        return subcell_id

    def update_neighbours(self):
        """
        :return:new neighbour list.
        """

        return self.__neighbour_list

    def get_neighbours(self, particle_pos):
        """
        Calculate which neighbours are around the particle.
        Calculates the distance between a particle and its neighbours
        :param particle_pos: 3d coordinates of single particle
        :return: array of neighbour particles and array with distances
        """
        neighbours = []
        neighbour_subcells = self.__get_neighbours_subcells(particle_pos)
        start_array = np.asarray(self.__start_index)
        #print("neighbours_subcells:", neighbour_subcells)

        #print("\n start_array:", start_array, "\n shape:", start_array.shape)
        for i in np.nditer(neighbour_subcells):
            index = start_array[i[0]]
            while index != 0:
                neighbours.append(index)
                index = self.__neighbour_list[index]

        positions = self.SystemState.positions()
        nb_length = np.shape(neighbours)[0]

        neighbours_distance = []
        new_neighbours = []
        for i in range(nb_length):
            index = neighbours[i]
            x_distance = particle_pos[0] - positions[index][0]
            y_distance = particle_pos[1] - positions[index][1]
            z_distance = particle_pos[2] - positions[index][2]

            distance = np.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
            # distance not further than cutoff radius:
            if distance <= self.SystemInfo.cutoff():
                neighbours_distance.append(distance)
                print("distance:", distance)
                new_neighbours.append(index)
                print("index: ", index)

        neighbours = new_neighbours
        print("neighbours:", neighbours)
        Result = collections.namedtuple("Neighbour_result", ["nb_pos", "nb_dist"])
        r = Result(nb_pos=neighbours, nb_dist=neighbours_distance)
        print(r)
        return r

    def __get_neighbours_subcells(self, particle_pos):
        """
        Issue: The particle counts itself so distances later should get rid of that!
        Call neighbour list from Class Neighbours
        :param particle_pos: 3d position of single particle
        :return: neighbours
        """

        m = self.__subcells_inrow  # makes code easier to read
        # get subcell id for surrounding subcells
        # (including particle_subcell)
        # find boundary boxes by using 3d coordinates

        # initialize subcell ids with subcell of particle
        subcells_id_3d = np.zeros((27, 3))
        subcells_id_3d[:] = self.__3d_subcell_id(particle_pos)

        #print("subcell_id_3d start:", subcells_id_3d[0])

        # First set of neighbour cells 0 to 8
        for cell in range(9):
            # x coordinates
            subcells_id_3d[cell][0] = math.floor((particle_pos[0]
                                                  - self.__subcell_length)
                                                 / self.__subcell_length)
            #print("subcell_id_3d for cell", cell, ":", subcells_id_3d[cell], "\n")

        # cells 18 to 26
        for cell in range(18, 27):
            # x coordinate
            subcells_id_3d[cell][0] = math.floor((particle_pos[0]
                                                  - self.__subcell_length)
                                                  / self.__subcell_length)

        # y-coordinates:
        for cell in [0, 1, 2, 9, 10, 11, 18, 19, 20]:
            subcells_id_3d[cell][1] = self.__cell_y(1, particle_pos)
        for cell in [6, 7, 8, 15, 16, 17, 24, 25, 26]:
            subcells_id_3d[cell][1] = self.__cell_y(0, particle_pos)
        #print("subcell_id_3d for cell:", subcells_id_3d)
        # z-coordinates:
        for cell in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
            subcells_id_3d[cell][2] = self.__cell_z(1, particle_pos)
        for cell in [2, 5, 8, 11, 14, 17, 20, 23, 26]:
            subcells_id_3d[cell][2] = self.__cell_z(0, particle_pos)
        #print("subcell_id_3d for cell:", subcells_id_3d)
        neighbour_subcells = [0] * 3 ** 3

        # transform 3d subcell ID to an integer
        for cell in range(27):

            # check if subcell is out of bounds(<0 or bigger than boxsize)
            for index in range(3):
                if subcells_id_3d[cell][index] < 0:
                    subcells_id_3d[cell][index] = m - 1
                if subcells_id_3d[cell][index] > self.__box_length:
                    subcells_id_3d[cell][index] = 0

            subcell_id = subcells_id_3d[cell][0] \
                         + (subcells_id_3d[cell][1] * m) \
                         + (subcells_id_3d[cell][2] * (m ** 2))

            neighbour_subcells[cell] = int(subcell_id)

        return neighbour_subcells

    #Following functions are for minor operations that occur several times.\n"
    #Functions: __3d_subcell_id, __cell_y, __cell_z

    def __3d_subcell_id(self, particle_pos):
        """
        Create subcell ID in 3d for a particle
        :param particle_pos: 3d coordinates of single particle
        :return: 3d-subcell ID
        """
        subcell_id_3d = np.zeros(3)
        for axis in range(3):
            subcell_id_3d[axis] = math.floor(particle_pos[axis]
                                            / self.__subcell_length)
        return subcell_id_3d

    def __cell_y(self, positive, particle_pos):
        """
        Calculates the subcell id for the y axis
        :param positive: boolean, if True the subcell id in positive
                        direction will be calculated
        :param particle_pos: 3d coordinates of single particle
        :return: subcell id for y axis
        """
        if positive == 0:
            y_id = math.floor((particle_pos[1] + self.__subcell_length) / self.__subcell_length)
        elif positive == 1:
            y_id = math.floor((particle_pos[1] - self.__subcell_length) / self.__subcell_length)
        return y_id

    def __cell_z(self, positive, particle_pos):
        """
        Calculates the subcell id for the y axis
        :param positive: boolean, if True the subcell id in positive
                         direction will be calculated
        :param particle_pos: 3d coordinates of a single particle
        :return: subcell id for y axis
        """
        if positive == 0:
            z_id = math.floor((particle_pos[2] + self.__subcell_length) / self.__subcell_length)
        elif positive == 1:
            z_id = math.floor((particle_pos[2] - self.__subcell_length) / self.__subcell_length)
        return z_id
