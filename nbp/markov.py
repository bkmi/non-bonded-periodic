import numpy as np
import scipy as sp
import scipy.stats
from .sysmodule import SystemState


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
        super().__init__(system, epsilon, sigma, n_steps, dt, length)
        self.__temperature = 0

    def __propose(self, cov):
        """Propose the next state, moves a single particle randomly with a 3d gaussian."""
        positions = self.__system.state().positions()
        particle = np.random.choice(positions.shape[0])
        proposal_positions = positions
        proposal_positions[particle] = sp.stats.multivariate_normal(np.zeros(3), cov * np.eye(3)).rvs()
        self.__proposal = SystemState(proposal_positions)

    def __check(self):
        position_energy = self.__system.state().energy()
        proposal_energy = self.__proposal.energy()
        if proposal_energy <= position_energy:
            return True
        else:
            return False

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to optimize"""
        cov = self.__system.info().char_length()
        self.__propose(cov)
        if self.__check():
            return self.__proposal
        else:
            return self.__system().state()


class Simulator(Actor):
    """The class that simulates."""
    def __init__(self, system, epsilon, sigma, n_steps=100, dt=0.001, length=1000):
        super().__init__(system, epsilon, sigma, n_steps, dt, length)

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        pass

    def __check(self):
        pass

    def __metropolis(self):
        """Proposes the new states"""
        pass
