import numpy as np
import math
import collections

class Neighbours:
    """
    This class is used to evaluate the position of
    particles to each others. The system must be big enough to
    contain at least 3 subcells per row.
    Private Functions:
    __init__, _create_subcells, _create_neighbours, _find_subcell,
    _get_neighbour_subcell, _3d_subcell_id, _cell_y, _cell_z
    Public Functions:
    update_neighbours, get_neighbours

    """
    def __init__(self, system_info, system_state):
        """
        Instantiates a new Object of class Neighbours
        :system: Instance of class System
        """
        self.SystemInfo = system_info
        self.SystemState = system_state
        self._box_length = self.SystemInfo.char_length()
        self._subcells_inrow = 1
        self._subcell_length = self._create_subcells()
        self._neighbour_list = self._create_neighbours()

    def _create_subcells(self):
        """
        Calculates the length of the subcells based on the skin radius.
        This helps to avoid having to update every time.
        :return: length of subcell
        """
        # define skin radius by using sigma.
        # The 3 is just an option that mostly works (?)
        _skin_radius = self.SystemInfo.sigma() * 3

        # calculate subcell size.
        while _skin_radius < self._box_length / self._subcells_inrow:
            self._subcells_inrow += 1

        print("Number of subcells per row:", self._subcells_inrow)
        subcell_length = self._box_length / self._subcells_inrow

        return subcell_length

    def _create_neighbours(self):
        """
        Creates the neighbours list by creating a head list which
        contains the starting index for the particles in a box.
        The neighbour list itself contains the index of the particles
        which belong to each box.
        :return: neighbour list
        """
        # get number of particles in the system
        particle_number = self.SystemState.positions().shape[0]

        # create list (head) for starting index of subcell.
        # **3 because we have 3 dimensions for 2 it would be **2.
        self._start_index = np.zeros(self._subcells_inrow**3)

        # create linked neighbour list
        self._neighbour_list = [-1] * particle_number

        # get positions of all particles
        positions = self.SystemState.positions()
        # print("positions:", positions)
        # Find list of particles each subcell contains
        for i in range(particle_number):

            subcell_id = self._find_subcell(positions[i])

            #print("This is i:", i)
            #print("This is the subcell ID:", subcell_id)
            try:
                self._neighbour_list[i] = int(self._start_index[subcell_id])
                self._start_index[subcell_id] = int(i)
            except IndexError:
                pass
                # print("IndexError: Index out of range in start_index", subcell_id)
        #print("created neighbour List:", self._neighbour_list,
         #    "\n start_index:", self._start_index)
        return self._neighbour_list

    def _find_subcell(self, position):
        """
        Private module of object neighbours. Finds the number of the
        subcell in which a particle is positioned.
        :param position: position of a particle
        :return: subcell_id
        """

        subcell_id_3d = [0, 0, 0]
        # calculate the subcell ID for each axis (x, y, z)
        for axis in range(3):
            try:
                subcell_id_3d[axis] = np.floor(position[axis]
                                                 / self._subcell_length)
            except OverflowError:
                pass

        # Use 3d subcell ID to get the 1d ID
        if (subcell_id_3d[0] < 0 or subcell_id_3d[1] < 0
                or subcell_id_3d[2] < 0):
            raise ValueError('subcell_id value is negative')
        else:
            m = self._subcells_inrow
            subcell_id = subcell_id_3d[0] \
                         + (subcell_id_3d[1] * m) \
                         + (subcell_id_3d[2] * (m ** 2))

        return subcell_id

    def update_neighbours(self):
        """
        :return:new neighbour list.
        """

        return self._neighbour_list

    def get_neighbours(self, particle_pos):
        """
        Calculate which neighbours are around the particle.
        Calculates the distance between a particle and its neighbours
        :param particle_pos: 3d coordinates of single particle
        :return: array of neighbour particles and array with distances
        """
        positions = self.SystemState.positions()
        neighbours = []           # neighbour positions
        neighbours_distance = []  # distance of particle to each neighbour
        new_neighbours = []       # only neighbours within cutoff radius
        neighbour_subcells = self._get_neighbours_subcells(particle_pos)
        # get starting positions for each subcell
        start_array = np.asarray(self._start_index)

        # get all particles from the neighbour subcells
        # np.nditer did not work for neighbour_subcells
        # for i in np.nditer(neighbour_subcells):
        # print("neighbour subcells:", neighbour_subcells)
        for i in range(27):
            i = neighbour_subcells[i]
            # print("i in loop:" , i)
            index = int(start_array[i])

            while index != 0:
                neighbours.append(index)
                index = int(self._neighbour_list[index])

        nb_length = np.shape(neighbours)[0]
        print("neighbour length:", nb_length)
        recent_neighbours = []
        # get distance from particle to neighbours
        for i in range(nb_length):
            index = neighbours[i]
            x_distance = particle_pos[0] - positions[index][0]
            y_distance = particle_pos[1] - positions[index][1]
            z_distance = particle_pos[2] - positions[index][2]

            # correct boundary subcells distance.
            # If only 2 subcells this will be have to caught somewhere else.
            # if self._subcells_inrow is 2:
            #    distance = np.sqrt(x_distance ** 2 + y_distance ** 2 + z_distance ** 2)
            #    print("Distance:", distance)
            #    if index not in recent_neighbours:
            #        recent_neighbours.append(index)
            #    else:
            #        x_distance = self._box_length - x_distance
            #        y_distance = self._box_length - y_distance
            #        z_distance = self._box_length - z_distance
            # else:

            l = 2*self._subcell_length  # max possible distance
            if x_distance > l:
                x_distance = self._box_length - x_distance
            if y_distance > l:
                y_distance = self._box_length - y_distance
            if z_distance > l:
                z_distance = self._box_length - z_distance


            distance = np.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
            # print("distance: ", distance)
            # distance no further than cutoff radius:
            if 0 < distance <= self.SystemInfo.cutoff():
                neighbours_distance.append(distance)
                new_neighbours.append(index)

        # overwrite neighbours with the correct ones.
        neighbours = new_neighbours

        # Create namedtuple for easy access of output
        # Result = collections.namedtuple("Neighbour_result", ["nb_pos", "nb_dist"])
        # r = Result(nb_pos=neighbours, nb_dist=neighbours_distance)

        return neighbours, neighbours_distance

    def set_neighbours(self, particle_pos):
        self._neighbours, self._neighbours_distance = self.get_neighbours(particle_pos)

    @property
    def nb_pos(self):
        return self._neighbours

    @property
    def nb_dist(self):
        return self._neighbours_distance

    def _get_neighbours_subcells(self, particle_pos):
        """
        Gets the subcell ID of the subcells which surround
        the subcell of the particle.
        To fulfill periodic boundary conditions, the subcell IDs
        are first calculated in 3D and subcell IDs which are
        out of bound are corrected.
        :param particle_pos: 3d position of single particle
        :return: neighbours
        """

        m = self._subcells_inrow  # makes code easier to read
        # initialize 3d subcell IDs with subcell of the particle
        subcells_id_3d = np.zeros((27, 3))
        subcells_id_3d[:] = self._3d_subcell_id(particle_pos)

        # First set of neighbour cells 0 to 8
        for cell in range(9):
            # x coordinates
            subcells_id_3d[cell][0] = np.floor((particle_pos[0]
                                                  - self._subcell_length)
                                                 / self._subcell_length)

        # cells 18 to 26
        for cell in range(18, 27):
            # x coordinate
            subcells_id_3d[cell][0] = np.floor((particle_pos[0]
                                                  - self._subcell_length)
                                                  / self._subcell_length)

        # y-coordinates:
        for cell in [0, 1, 2, 9, 10, 11, 18, 19, 20]:
            subcells_id_3d[cell][1] = self._cell_y(1, particle_pos)
        for cell in [6, 7, 8, 15, 16, 17, 24, 25, 26]:
            subcells_id_3d[cell][1] = self._cell_y(0, particle_pos)
        # z-coordinates:
        for cell in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
            subcells_id_3d[cell][2] = self._cell_z(1, particle_pos)
        for cell in [2, 5, 8, 11, 14, 17, 20, 23, 26]:
            subcells_id_3d[cell][2] = self._cell_z(0, particle_pos)

        # initialize neighbour subcells (3x3x3 cube)
        neighbour_subcells = [0] * 3 ** 3

        # transform 3d subcell ID to an integer
        for cell in range(27):

            # check if subcell is out of bounds(<0 or bigger than boxsize)
            for index in range(3):
                if subcells_id_3d[cell][index] < 0:
                    subcells_id_3d[cell][index] = m - 1
                if subcells_id_3d[cell][index] > m - 1:
                    subcells_id_3d[cell][index] = 0

            subcell_id = subcells_id_3d[cell][0] \
                         + (subcells_id_3d[cell][1] * m) \
                         + (subcells_id_3d[cell][2] * (m ** 2))

            neighbour_subcells[cell] = int(subcell_id)

        return neighbour_subcells

# Following functions are for minor operations that occur several times.\n"
# Functions: _3d_subcell_id, _cell_y, _cell_z

    def _3d_subcell_id(self, particle_pos):
        """
        Create subcell ID in 3d for a particle
        :param particle_pos: 3d coordinates of single particle
        :return: 3d-subcell ID
        """
        subcell_id_3d = np.zeros(3)
        for axis in range(3):
            subcell_id_3d[axis] = np.floor(particle_pos[axis]
                                            / self._subcell_length)
        return subcell_id_3d

    def _cell_y(self, positive, particle_pos):
        """
        Calculates the subcell id for the y axis
        :param positive: boolean, if True the subcell id in positive
                        direction will be calculated
        :param particle_pos: 3d coordinates of single particle
        :return: subcell id for y axis
        """
        if positive == 0:
            y_id = np.floor((particle_pos[1] + self._subcell_length) / self._subcell_length)
        else:
            y_id = np.floor((particle_pos[1] - self._subcell_length) / self._subcell_length)
        return y_id

    def _cell_z(self, positive, particle_pos):
        """
        Calculates the subcell id for the y axis
        :param positive: boolean, if True the subcell id in positive
                         direction will be calculated
        :param particle_pos: 3d coordinates of a single particle
        :return: subcell id for y axis
        """
        if positive == 0:
            z_id = np.floor((particle_pos[2] + self._subcell_length) / self._subcell_length)
        else:
            z_id = np.floor((particle_pos[2] - self._subcell_length) / self._subcell_length)
        return z_id
