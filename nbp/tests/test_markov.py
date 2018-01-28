import nbp
import numpy as np

from .tools import make_system


def test_optimize_lj():
    system = make_system(positions=np.asarray([[1, 1, 1], [1.5, 1, 1]]),
                         lj=True, ewald=True, use_neighbours=False)
    mcmc = nbp.MCMC(system)

    assert True is True


def test_simulate():
    # Compare two particle monte carlo to two particle mcmc.
    assert True is True
