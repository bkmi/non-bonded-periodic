import numpy as np
import scipy as sp
import scipy.stats
import nbp


class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self._system = system

    def optimize(self, max_steps, d_energy_tol=1e-6):
        """Optimize from the last system state."""
        optimizer = Optimizer(self._system)
        old_energy = 0
        for i in range(max_steps):
            new_state, new_energy = optimizer.act()
            self._system.update_state(new_state)
            if abs(new_energy - old_energy) < d_energy_tol:
                break
            else:
                old_energy = new_energy
        return self._system

    def simulate(self, steps, temperature):
        """Simulate from the last system state."""
        simulator = Simulator(self._system)
        for i in range(steps):
            self._system.update_state(simulator.act(temperature))
        return self._system


class Optimizer:
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system):
        self._system = system
        self._proposal = None

    def _propose(self, cov):
        """Propose the next state, moves a single particle randomly with a 3d gaussian.

        returns proposal state, proposal_energy"""
        positions = self._system.state().positions()
        particle = np.random.choice(positions.shape[0])
        proposal_positions = positions
        proposal_positions[particle] = sp.stats.multivariate_normal(np.zeros(3), cov * np.eye(3)).rvs()
        proposal_state = nbp.SystemState(proposal_positions)
        return proposal_state, proposal_state.energy()

    @staticmethod
    def _check(orig_energy, proposal_energy):
        if proposal_energy <= orig_energy:
            return True
        else:
            return False

    def act(self, temperature=0):
        """Overriding of the function act of the Actor in order for it to optimize"""
        cov = self._system.info().char_length()
        orig_energy = self._system.energy()
        self._proposal, proposal_energy = self._propose(cov)
        if self._check(orig_energy, proposal_energy):
            return self._proposal, proposal_energy
        else:
            return self._system().state(), orig_energy


class Simulator:
    """The class that simulates."""
    def __init__(self, system):
        self._system = system

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        return None

    def _check(self):
        pass
