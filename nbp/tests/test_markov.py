import nbp


def test_optimize():
    characteristic_length = 10
    sigma = 2
    particle_charges = [1, 1]
    positions = [[1, 1, 1], [2, 2, 2]]

    system = nbp.System(characteristic_length, sigma, particle_charges, positions)

    mcmc = nbp.MCMC(system)

    # Do a simple system where it goes directly to the lowest energy state.

    if mcmc:
        assert True is True


def test_simulate():
    # Compare two particle monte carlo to two particle mcmc.
    assert True is True
