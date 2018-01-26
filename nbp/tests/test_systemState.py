from unittest import TestCase
import numpy as np
import nbp

from .tools import make_system


class TestSystemState(TestCase):

    @staticmethod
    def make_system(sigma=[0.5]):
        box_length = 10
        particle_count = 50
        random_positions = box_length * np.random.rand(particle_count, 3)
        random_charges = np.random.rand(particle_count, 1)
        system = nbp.System(box_length, sigma, random_charges, random_positions)
        return system

    def test_potential_lj_one(self):
        """Tests if given distance 1.0 and sigma 1.0,  the result is zero"""
        system = self.make_system([1.0])
        self.assertEqual(0, system.state()._potential_lj(1.0, 1.0))
        return

    def test_potential_lj_two(self):
        system = self.make_system()
        sigma = np.random.rand()
        distance = np.random.rand()
        q = (sigma / distance) **6
        q = 4.0 * system.info().epsilon0() * q * (q - 1)
        self.assertEqual(q, system.state()._potential_lj(distance, sigma))

    def test_potential(self):
        """TODO Problems in neighbours before the potential itself is called"""
        self.assertTrue()
