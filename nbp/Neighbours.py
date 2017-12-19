from SystemModule import System

class Neighbours:

    def __init__(self, system):
        """
        Instantiates a new Object of class Neighbours
        :system: Instance of class System
        """
        self.System = system
        self.__neighbour_list = self.create_neighbours()



    def __create_subcells(self):

        # define skin radius by using sigma.
        # The 3 is just an option that mostly works (?)
        __skin_radius = self.System.info().sigma * 3
        __box_dim = self.System.state.positions.shape()

        self.__subcell_number = 1 # initialize

        # calculate subcell size
        while __skin_radius < __box_dim/self.__subcell_number:
            self.__subcell_number += 1

        __subcell_length = box_dim/self.__subcell_number # do we need this?

        pass


    def __create_neighbours(self):

        __particle_number = self.System.state().positions().shape[0]
        # create list for starting index of subcell
        __start_index = [0] * self.__subcell_number^3
        # create neighbour list
        self.__neighbour_list = []
        # get positions
        __positions = self.System.state.positions()

        for i in range(0, __particle_number - 1):

            self.__neighbour_list[i] = __start_index[icell]
            __start_index[icell] = i

        pass


    def update_neighbours(self):
        """
        :return:new neighbour list.
        """

        return __neighbour_list