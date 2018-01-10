import numpy as np
import scipy as sp
import scipy.stats
from .sysmodule import SystemState


class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self.__system = system

    def optimize(self, steps):
        """Optimize from the last system state."""
        actor = Optimizer(self.__system)
        for i in range(steps):
            self.__system.update_state(actor.act())
        return self.__system

    def simulate(self, steps, temperature):
        """Simulate from the last system state."""
        actor = Simulator(self.__system)
        for i in range(steps):
            self.__system.update_state(actor.act(temperature))
        return self.__system


class Actor:
    """Methods wrapper for the proposal and acceptance steps in MCMC."""
    def __init__(self, system):
        self.__system = system

    def act(self, temperature):
        """Interface for the function that is called by the MCMC to optimize or
        simulate, accordingly to the phase is it in"""
        raise NotImplementedError('Actor is merely an interface, choose an implementation of Actor.')

    def __check(self):
        """Checks if the new proposed system is to be accepted"""
        raise NotImplementedError('Actor is merely an interface, choose an implementation of Actor.')


class Optimizer(Actor):
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system):
        super().__init__(system)

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

    def act(self, temperature=0):
        """Overriding of the function act of the Actor in order for it to optimize"""
        cov = self.__system.info().char_length()
        self.__propose(cov)
        if self.__check():
            return self.__proposal
        else:
            return self.__system().state()


class Simulator(Actor):
    """The class that simulates."""
    def __init__(self, system):
        super().__init__(system)

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        return None

    def __check(self):
        pass
