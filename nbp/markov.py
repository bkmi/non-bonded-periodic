import numpy as np
import scipy as sp
import scipy.stats
from .sysmodule import SystemState


class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self.__system = system

    def optimize(self, max_steps, d_energy_tol=1e-6):
        """Optimize from the last system state."""
        optimizer = Optimizer(self.__system)
        old_energy = 0
        for i in range(max_steps):
            new_state, new_energy = optimizer.act()
            self.__system.update_state(new_state)
            if abs(new_energy - old_energy) < d_energy_tol:
                break
            else:
                old_energy = new_energy
        return self.__system

    def simulate(self, steps, temperature):
        """Simulate from the last system state."""
        simulator = Simulator(self.__system)
        for i in range(steps):
            self.__system.update_state(simulator.act(temperature))
        return self.__system


class Optimizer:
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system):
        self.__system = system
        self.__proposal = None

    def __propose(self, cov):
        """Propose the next state, moves a single particle randomly with a 3d gaussian.

        returns proposal state, proposal_energy"""
        positions = self.__system.state().positions()
        particle = np.random.choice(positions.shape[0])
        proposal_positions = positions
        proposal_positions[particle] = sp.stats.multivariate_normal(np.zeros(3), cov * np.eye(3)).rvs()
        proposal_state = SystemState(proposal_positions)
        return proposal_state, proposal_state.energy()

    @staticmethod
    def __check(orig_energy, proposal_energy):
        if proposal_energy <= orig_energy:
            return True
        else:
            return False

    def act(self, temperature=0):
        """Overriding of the function act of the Actor in order for it to optimize"""
        cov = self.__system.info().char_length()
        orig_energy = self.__system.energy()
        self.__proposal, proposal_energy = self.__propose(cov)
        if self.__check(orig_energy, proposal_energy):
            return self.__proposal, proposal_energy
        else:
            return self.__system().state(), orig_energy


class Simulator:
    """The class that simulates."""
    def __init__(self, system):
        self.__system = system

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        return None

    def __check(self):
        pass
