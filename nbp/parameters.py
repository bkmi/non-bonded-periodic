import numpy as np
import nbp

class Parameters:
    """
    This class contains parameters which are derived
    through statistical processes.
    """

    def __init__(self, p=5.0):
        """
        Initialize the sigma needed for the gaussian distribution
        in the long-range energy calculations in the ewald sum.

        :param p: float (default: 5.0)
                  the higher p is the higher is the accuracy
        """
        self._r_cutoff = nbp.SystemInfo.cutoff()
        self._gauss_sigma = self._get_sigma(p)
        self._k_cutoff = self._get_k_cutoff(p)
        pass

    def _get_sigma(self, p):
        """
        Calculate the sigma that is needed in the gaussian distribution
        of the long-range ewald sum. The default for p is 5.0 which will
        give an estimated error accuracy > 0.01.

        :param p: float
                  defines the accuracy
        :return: float
                 sigma for gaussian distribution  """
        self._gauss_sigma = self._r_cutoff/np.sqrt(2*p)
        return self._gauss_sigma

    def gauss_sigma(self):
        """
        Returns the sigma for the gaussian distribution needed in the
        long-range ewald sum.

        :return: float
                 sigma for gaussian distribution
        """
        return self._gauss_sigma

    def _get_k_cutoff(self, p):
        """
        Calculate the cutoff for the k vector in the long-range ewald
        sum. Based on the accuracy parameter and the cutoff of the normal
        radius.

        :param p: float
                   defines the accuracy
        :return: float
                 gives cutoff for k-vector"""
        self._k_cutoff = 2 * p/self._r_cutoff
        return self._k_cutoff

    def k_cutoff(self):
        """ Returns the cutoff of the k-vector needed in the
        long-range ewald sum.

        :return: float
                 gives cutoff for k-vector"""

        return self._k_cutoff
