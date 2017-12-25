import numpy as np


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
        pass

    def __propose(self):
        """Propose the next state"""
        last_state = self.__system.states[-1].positions
        next_state = last_state + self.__system.electrostatics.forces + scipy.stats.multivariate_normal(
            numpy.zeros(last_state.size), numpy.eye(last_state.shape[1])).rvs(1)
        return next_state

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
