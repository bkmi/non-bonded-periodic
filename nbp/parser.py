import numpy as np


class Parser:

    """The parser that converts the input data to data usable by the simulator

    Attributes:
        box:    ndarray
                3-dimensional array containing the dimensions of the box
        positions:  ndarray
                    n-dimensional array containing all the 3D positions of the particles
        types:  ndarray
                n-dimensional array containing the type of each particle
        parameters: dict
                    [sigma, epsilon and mass] for each particle type
        """

    __SIGMA = 0
    __EPSILON = 1
    __MASS = 2
    __CHARGE = 3

    def __init__(self, file='sodium-chloride-example.npz'):
        """The initialisation function.

        :param
            file:   string, optional (default = 'sodium-chloride-example.npz'
                    the file containing all the system static and initial information
        """

        with np.load(file) as fh:
            self.box = fh['box']
            self.positions = fh['positions']
            self.types = fh['types']
            self.parameters = fh['parameters'].item()
        pass

    def parse(self):
        """Parsee the content of the file into variables readable by the software.

        Unless stated, all the attributes in the form of an array or a matrix follow the following convention:
        information[i] contains information about the particle i
        pairInformation[i][j] contains information about the relation between particle i and j (e.g. the distance).

        :return
            dict
            a dictionary with the following keys:
                ch_length:  float
                            the characteristic length of the box.
                pos:    ndarray
                        a n-dimensional array containing the 3D positions of the particles.
                sigma:  ndarray
                        a n-dimensional array containing for each particle its sigma.
                epsilon:    ndarray
                            a n-dimensional array containing for each particle its epsilon.
                mass:   ndarray
                        a n-dimensional array containing for each particle its mass.
                type:   list
                        a n-dimensional array containing for each particle its type.
        """

        sigma = []
        epsilon = []
        mass = []
        charge = []
        for i in range(len(self.types)):
            par = self.parameters[self.types[i]]
            sigma.append(par[Parser.__SIGMA])
            epsilon.append(par[Parser.__EPSILON])
            mass.append(par[Parser.__MASS])
            charge.append(par[Parser.__CHARGE])

        return {'ch_length': max(self.box), 'pos': np.asarray(self.positions), 'sigma': np.asarray(sigma),
                'epsilon': np.asarray(epsilon), 'mass': np.asarray(mass), 'charge' : np.asarray(charge),
                'type': np.asarray(self.types), 'param': self.parameters}

    pass
