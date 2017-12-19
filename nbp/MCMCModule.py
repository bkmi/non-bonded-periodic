import numpy as np
import scipy.stats
from .SystemModule import *
from .unitconvert import cart_to_spher, spher_to_cart

class MCMC:
    """An class which applies actor to a System instance and updates it to the next step in MCMC."""
    def __init__(self, system, actor):
        self.__system = system
        self.__actor = actor

    def propose_next_state(self):
        """Propose the next state by calling the actor"""
        pass


class Actor:
    """Methods wrapper for the proposal and acceptance steps in MCMC."""
    def __init__(self, system, epsilon, sigma, n_steps=100, dt=0.001, length=1000):
        self.__system = system
        self.__epsilon = epsilon
        self.__sigma = sigma
        self.__n_steps = n_steps
        self.__dt = dt
        self.__length = length
        self.__previous_electrostatics = None

    def act(self, temperature):
        """Interface for the function that is called by the MCMC to optimize or
        simulate, accordingly to the phase is it in"""
        raise NotImplementedError('Actor is merely an interface, choose an implementation of Actor.')

    def __check(self):
        """Checks if the new proposed system is to be accepted"""
        raise NotImplementedError('Actor is merely an interface, choose an implementation of Actor.')


class Optimizer(Actor):
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system, epsilon, sigma, n_steps=100, dt=0.001, length=1000):
        super(Optimizer).__init__(system, epsilon, sigma, n_steps, dt, length)
        self.__temperature = 0
        pass

    def __propose(self):
        """Propose the next state"""
        pass

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to optimize"""
        pass

    def __check(self):
        pass

    def __gradient_descent(self):
        """Proposes the new states"""
        pass


class Simulator(Actor):
    """The class that simulates."""
    def __init__(self, temperature):
        super(Simulator, self).__init__()
        self.__temperature = temperature

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        new_positions = self.__metropolis_step(self.system.state.positions())
        new_electrostatics = Electrostatic(new_positions)
        new_state = SystemState(new_positions, new_electrostatics)
        if self.__check(new_state):
            self.system.update_state(new_state)
        else:
            self.system.update_state(self.system.state())
        pass

    def __check(self, new_state):
        pass

    def __metropolis_step(self, positions):
        """Proposes the new states"""
        k_b = 1.38064852e-23    # Boltzmann constant in [J/K]
        n = len(positions)      # number of particles
        v_avg = [np.sqrt(2*k_b*self.temperature/particle.particle_mass) for particle in self.system.info().particles]
        s_avg = [v*self.__dt for v in v_avg]
        # <s_avg> can be used to change the positions of the particles. Travel distances will be normally distributed around the <s_avg>
        r_new = np.copy(positions)
        indices = np.random.randint(0, n, size=np.random.randint(np.floor(n * 0.1), np.floor(n * 0.5)))
        # an array of random particle indices whose positions will be changed (between 10 and 50 % of the particles)
        r_new[indices] = list(map(cart_to_spher(), r_new[indices]))
        # map artesian coordinates to spherical
        r_new[indices] += [
            np.array([s_avg * np.random.random(),   # random between 0 and s_avg
                      np.pi * np.random.random(),   # random between 0 and pi
                      2 * np.pi * np.random.random()])  # random b/w 0 and 2pi
            for _ in range(len(indices))
        ]
        # suggest new positions
        r_new[indices] = list(map(spher_to_cart(), r_new[indices]))
        # map back to cartesian coordinates
        return r_new

    @property
    def temperature(self):
        return self.__temperature
