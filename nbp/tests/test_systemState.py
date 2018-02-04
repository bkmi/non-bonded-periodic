from unittest import TestCase
import numpy as np
import nbp
from nbp.tests.tools import make_system


class TestSystemState(TestCase):

    # def test_potential_lj_one(self):
    #     """Tests if given distance 1.0 and sigma 1.0,  the result is zero"""
    #     system = make_system(lj=True, ewald=False, use_neighbours=False, particle_count=2,
    #                          positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    #     self.assertEqual(np.zeros((2, 2)), system.state()._potential_lj())

    def test_potential_lj_two(self):
        """Tests if the potential works with only 1 particle"""
        sigma_test = np.matrix(np.random.rand())
        epsilon_test = np.matrix(np.random.rand())
        system = make_system(lj=True, ewald=False, use_neighbours=False, particle_count=1, sigma=sigma_test,
                             epsilon_lj=epsilon_test)
        self.assertEqual(system.state().potential_lj(), [0])
