import numpy as np
import scipy as sp
import scipy.stats
import nbp


class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self._system = system

    def optimize(self, max_steps, d_energy_tol=1e-6, no_progress_break=10):
        """Optimize from the last system state."""
        optimizer = Optimizer(self._system)
        energies = []
        additional_states = []
        for i in range(max_steps):
            new_state, new_energy = optimizer.act()
            additional_states.append(new_state)
            energies.append(new_energy)
            if len(energies) > no_progress_break and np.all(
                    np.less(np.abs(np.asarray(energies)[-no_progress_break:] - new_energy), d_energy_tol)):
                break
        return additional_states

    def simulate(self, steps, temperature):
        """Simulate from the last system state."""
        simulator = Simulator(self._system)
        additional_states = []
        for i in range(steps):
            additional_states.append(simulator.act(temperature))
        return additional_states


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

    def act(self):
        """Overriding of the function act of the Actor in order for it to optimize"""
        cov = self._system.info().char_length()
        orig_energy = self._system.energy()
        self._proposal, proposal_energy = self._propose(cov)
        if self._check(orig_energy, proposal_energy):
            return self._proposal, proposal_energy
        else:
            return self._system.state(), orig_energy


class Simulator:
    """The class that simulates."""
    def __init__(self, system):
        self._system = system

    def act(self, temperature):
        """Overriding of the function act of the Actor in order for it to simulate"""
        cov = 1  # TODO scale covariance
        num_particles = len(self._system.state().positions())
        indices_toMove = list(set(np.random.randint(num_particles, size=np.random.randint(1, num_particles))))
        proposal_state = self._metropolis(indices_toMove, cov)
        if self._check(proposal_state, temperature):
            return proposal_state
        else:
            return self._system.state()

    def _check(self, state, temperature):
        """Checks for the acceptance of a proposal state"""
        beta = 1  # TODO: Calculate beta (with dim=1) used in Boltzmann Factor
        energy_prev = self._system.states()[-1].energy()
        energy_curr = state.energy()
        p_acc = beta * (energy_curr - energy_prev)
        if np.random.random() <= p_acc:
            return True
        else:
            return False

    def _metropolis(self, indices, cov):
        """Proposes the new states"""
        new_positions = np.copy(self._system.state().positions())
        new_positions[indices] = np.array(
            [sp.stats.multivariate_normal(each, cov=cov).rvs().tolist() for each in new_positions[indices]])
        proposal_state = nbp.SystemState(new_positions)
        return proposal_state
