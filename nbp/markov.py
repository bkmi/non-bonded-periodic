import numpy as np
from scipy import stats
from .sysmodule import SystemState

class MCMC:
    """A class which applies actor to a System instance and updates it to the next step in MCMC."""
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

    def __check(self, state, temperature):
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
        next_state = last_state + self.__system.electrostatics.forces + stats.multivariate_normal(
            np.zeros(last_state.size), np.eye(last_state.shape[1])).rvs(1)
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
        super(Simulator, self).__init__(system, epsilon, sigma, n_steps, dt, length)

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        cov = 1     #TODO scale covariance
        num_particles = len(self.system.state().positions())
        indices_toMove = list(set(np.random.randint(num_particles, size=np.random.randint(1, num_particles))))
        proposal_state = self.__metropolis(indices_toMove, cov)
        if self.__check(proposal_state, temperature):
            self.system.update_state(proposal_state)
        else:
            self.system.update_state(self.system.state())

    def __check(self, state, temperature):
        """Checks for the acceptance of a proposal state"""
        beta = 1  # TODO: Calculate beta (with dim=1) used in Boltzmann Factor
        energy_prev = self.system.states()[-1].energy()
        energy_curr = state.energy()
        p_acc = beta * (energy_curr-energy_prev)
        if np.random.random() <= p_acc:
            return True
        else:
            return False

    def __metropolis(self, indices, cov):
        """Proposes the new states"""
        new_positions = np.copy(self.system.state().positions())
        new_positions[indices] = np.array([stats.multivariate_normal(each, cov=cov).rvs().tolist() for each in new_positions[indices]])
        proposal_state = SystemState(new_positions)
        return proposal_state
