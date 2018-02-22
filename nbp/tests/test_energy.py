import numpy as np
import numpy.testing as npt

import nbp
from nbp.tests.tools import make_system


def stnd_energy():
    # characteristic_length = 20
    # positions = (np.asarray([[1, 0, -2 ** (-1 / 2)],
    #                          [-1, 0, -2 ** (-1 / 2)],
    #                          [0, 1, 2 ** (-1 / 2)],
    #                          [0, -1, 2 ** (-1 / 2)]]) + characteristic_length / 2)
    # positions = positions,
    system = make_system(lj=True, ewald=False, use_neighbours=False)
    pos = system.state().positions()
    pairwise_distance_vec = pos[None, :, :] - pos[:, None, :]
    wrapped_pairwise_distance_vec = np.apply_along_axis(
        lambda x: nbp.periodic_wrap_corner(x, system.info().char_length()), -1, pairwise_distance_vec)
    wrapped_pairwise_distance = np.linalg.norm(wrapped_pairwise_distance_vec, axis=-1)

    lj_energy = np.zeros_like(wrapped_pairwise_distance)
    for ind, val in np.ndenumerate(wrapped_pairwise_distance):
        sigma_eff = np.mean([system.info().sigma()[ind[0]], system.info().sigma()[ind[1]]])
        epsilon_eff = np.linalg.norm([system.info().epsilon_lj()[ind[0]] ** 2, system.info().epsilon_lj()[ind[1]] ** 2])
        q = (sigma_eff / val) ** 6
        lj_energy[ind] = 4.0 * epsilon_eff * (q * (q - 1))

    actual_energy = np.sum(np.triu(lj_energy, k=1))

    return pos, actual_energy


def test_LJneighbours_vs_stnd():
    pos, actual_energy = stnd_energy()
    system = make_system(positions=pos, lj=True, ewald=False, use_neighbours=True)

    npt.assert_approx_equal(system.state().energy(),
                            float(actual_energy),
                            significant=2)


def test_LJ_versus_stnd():
    pos, actual_energy = stnd_energy()
    system = make_system(positions=pos, lj=True, ewald=False, use_neighbours=False)

    npt.assert_approx_equal(system.state().energy(),
                            float(actual_energy),
                            significant=5)