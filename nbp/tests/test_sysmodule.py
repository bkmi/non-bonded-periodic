import nbp
import numpy as np


class test_potential_lj():

    def test_sigma_exception(self):
        distance = np.random.random()
        sigma = -1.0
        self.assertRaises(AttributeError, nbp.SystemState._potential_lj(distance, sigma))

    def test_distance_exception(self):
        distance = 0
        sigma = np.random.random()
        self.assertRaises(AttributeError, nbp.SystemState._potential_lj(distance, sigma))

    def test_one(self):
        """Testes if given distance 1.0 and sigma 1.0,  the result is zero"""
        system = nbp.System(1.0, 1.0, 0.0, 1.0)
        self.assertEqual(0, system.state()._potential_lj(1.0, 1.0))

