import numpy as np
import scipy as sp
import scipy.stats
import nbp


class MCMC:
    """An class which applies actor to a System instance and doing MCMC for a set of steps."""
    def __init__(self, system):
        self._system = system

    def optimize(self, max_steps=500, cov=None, d_energy_tol=1e-6, no_progress_break=25, num_particles=0.05,
                 drop_intermediate_states=True):
        """Optimize from the last system state."""
        optimized_system = self._system
        optimizer = Optimizer(optimized_system)
        energies = []
        if cov is None:
            cov = optimized_system.info().cutoff()/2**7
        for i in range(max_steps):
            new_state, new_energy = optimizer.act(cov, num_particles=num_particles)
            optimized_system.update_state(new_state)
            energies.append(optimized_system.state().energy())
            if len(energies) > no_progress_break and np.all(
                    np.less(np.abs(np.asarray(energies)[-no_progress_break:] - new_energy), d_energy_tol)):
                break

        if drop_intermediate_states:
            optimized_system._systemStates = [optimized_system.state()]

        return optimized_system

    def simulate(self, steps, temperature, verbose=False):
        """Simulate from the last system state."""
        simulator = Simulator(self._system)
        for i in range(steps):
            self._system.update_state(simulator.act(temperature))

        if verbose:
            print(simulator.accepted_number)
        return self._system.state()


class Optimizer:
    """The class that optimizes the system to temperature 0"""
    def __init__(self, system):
        self._system = system
        self._proposal = None

    def _propose(self, cov, num_particles=None):
        """Propose the next state, moves a num_particles randomly with a 3d gaussian.
        returns proposal state, proposal_energy"""
        positions = self._system.state().positions()
        if isinstance(num_particles, float) and num_particles <= 1:
            num_particles = int(positions.shape[0] * num_particles)
        elif isinstance(num_particles, int) and num_particles <= positions.shape[0]:
            pass
        else:
            raise ValueError('num_particles must be a percentage (float) or a number of particles (int).')
        particles = np.random.choice(positions.shape[0], size=num_particles, replace=False)
        proposal_positions = positions
        proposal_positions[particles] = positions[particles] + sp.stats.multivariate_normal(
            np.zeros(3), cov * np.eye(3)).rvs(num_particles)
        proposal_state = nbp.SystemState(proposal_positions, self._system)
        return proposal_state, proposal_state.energy()

    @staticmethod
    def _check(orig_energy, proposal_energy):
        if proposal_energy <= orig_energy:
            return True
        else:
            return False

    def act(self, cov, num_particles=0.25):
        """Propose and check then return a new state."""
        orig_energy = self._system.state().energy()
        self._proposal, proposal_energy = self._propose(cov, num_particles=num_particles)
        if self._check(orig_energy, proposal_energy):
            return self._proposal, proposal_energy
        else:
            return self._system.state(), orig_energy


class Simulator:
    """Simulation class"""
    def __init__(self, system):
        """
        :param system: (obj: nbp.System): An instance of the <nbp.System> class
        """
        self._system = system
        self._accepted_number = 0

    def act(self, temperature):
        """
        A method for returning proposal states in the MCMC
        Args:
            temperature (float): Bath temperature in Kelvin

        Returns:
            proposal_state (obj: nbp.SystemState)
        """
        cov = self._system.info().sigma()[0]/500
        print(cov)
        num_particles = len(self._system.state().positions())
        indices_to_move = list(set(np.random.choice(np.arange(num_particles), size=int(np.ceil((0.05*num_particles))))))
        proposal_state, proposal_energy = self._metropolis(indices_to_move, cov)
        starting_energy = self._system.state().energy()
        print("prop {} start {}".format(proposal_energy, starting_energy))
        if self._check(temperature, proposal_energy, starting_energy):
            self._accepted_number += 1
            return proposal_state
        else:
            return self._system.state()

    @staticmethod
    def _check(temperature, energy_prop, energy_prev):
        """A method for checking for the acceptance of the proposed state.

            :param temperature: (float)
                The temperature of the heat bath

            :param energy_prop: (float)
                The energy of the proposed state

            :param energy_prev: (float)
                The energy of the starting state

            :return bool: Returns True if the state is accepted and False if rejected
        """
        _k_b = 1     # Boltzmann constant in Gaussian units [eV/K]
        beta = 1.380e-26#1/(_k_b*temperature)     # reciprocal of thermal energy in [eV]
        p_acc = np.min((1, np.exp(-beta * (energy_prop - energy_prev))))
        if np.random.random() <= p_acc:
            return True
        else:
            return False

    def _metropolis(self, indices, cov):
        """Proposes the new states

            :param indices: (list)
                A list containing the indices of moving particles
            :param cov: (float)
                Covariance for defining the multivariate normal
            :return (proposal state, proposal energy):  (nbp.State, float)
                An nbp.State instance as well as its energy
        """
        new_positions = np.copy(self._system.state().positions())
        new_positions[indices] += np.array(sp.stats.multivariate_normal(np.zeros(3), cov=cov).rvs(len(indices)).tolist())
        proposal_state = nbp.SystemState(new_positions, self._system)
        proposal_energy = proposal_state.energy()
        return proposal_state, proposal_energy

    @property
    def accepted_number(self):
        """"""
        return self._accepted_number
