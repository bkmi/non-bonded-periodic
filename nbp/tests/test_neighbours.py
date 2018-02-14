import nbp
import numpy as np


def make_system(characteristic_length=10,
                sigma=None, epsilon_lj=None, particle_charges=None, positions=None, particle_count=None,
                lj=True, ewald=True, use_neighbours=False):
    if particle_count is None:
        if particle_charges is not None:
            particle_count = np.asarray(particle_charges).shape[0]
        elif positions is not None:
            particle_count = np.asarray(positions).shape[0]
        else:
            particle_count = 50

    if not sigma:
        sigma = np.ones((particle_count, 1))
    if not epsilon_lj:
        epsilon_lj = np.ones((particle_count, 1))

    if particle_charges is None:
        particle_charges = np.random.rand(particle_count, 1)
    if positions is None:
        positions = characteristic_length * np.random.rand(particle_count, 3)

    system = nbp.System(characteristic_length, sigma, epsilon_lj, particle_charges, positions,
                        lj=lj, ewald=ewald, use_neighbours=use_neighbours)
    return system


def test_no_self_in_neighbours():
    system = make_system()

    self_in_neighbours = []
    for particle, position in enumerate(system.state().positions()):
        result = system.state().neighbours().get_neighbours(particle)
        if particle in result.nb_ID:
            self_in_neighbours.append(True)
        else:
            self_in_neighbours.append(False)

    assert any(self_in_neighbours) is False


def test_communicativity():
    system = make_system()

    for particle, position in enumerate(system.state().positions()):
        result = system.state().neighbours().get_neighbours(particle)
        nb_ID = result.nb_ID
        nb_dist = result.nb_dist
        print('Neighbours of particle {0}: {1}'.format(particle, nb_ID))
        print('Distances from particle {0}: {1}'.format(particle, nb_dist))
        for neighbour_index, neighbour in enumerate(nb_ID):
            neighbour_result = system.state().neighbours().get_neighbours(neighbour)
            neighbour_neighbours = neighbour_result.nb_ID
            neighbour_dist = neighbour_result.nb_dist
            print('Neighbours of particle {0}: {1}'.format(neighbour, neighbour_neighbours))
            print('Distances from particle {0}: {1}'.format(neighbour, neighbour_dist))

            if particle not in neighbour_neighbours:
                assert particle in neighbour_neighbours

            if neighbour_dist[neighbour_neighbours.index(particle)] != nb_dist[neighbour_index]:
                assert neighbour_dist[neighbour_neighbours.index(particle)] == nb_dist[neighbour_index]
        assert True


if 0:
    def setup_neighbours(positions, particle_count, char_length, x):
        charges = np.random.rand(particle_count, 1)
        sigma = np.ones((particle_count, 1))
        epsilon_lj = np.ones((particle_count, 1))
        system = nbp.System(characteristic_length=char_length,
                            sigma=sigma, epsilon_lj=epsilon_lj, particle_charges=charges, positions=positions,
                            lj=True, ewald=True, use_neighbours=True)
        neighbours = nbp.Neighbours(system.info(), system.state(), system, verbose=False)
        neighbours_list = neighbours.get_neighbours(x)

        return neighbours_list


    positions_set1 = 4 * np.random.random_sample((100, 3))
    nb_result = setup_neighbours(positions_set1, len(positions_set1), 5, 8)

    # Call the named tuple and show the positions and the distance
    print("position result 1:", nb_result.nb_ID)
    print("distance result 1:",  nb_result.nb_dist)
    print("------------------End Test 1 ----------------------------- \n")

    # Test if a real neighbour is seen as one.
    positions_set2 = [[0, 0, 0], [1, 1.5, 1.2], [11, 11, 10]]
    positions_set2 = np.asarray(positions_set2)
    nb_result2 = setup_neighbours(positions_set2, len(positions_set2), 10, 2)

    # Call the named tuple and show the positions and the distance
    print("position result 2:", nb_result2.nb_ID)
    print("distance result 2:",  nb_result2.nb_dist)
    print("--------------End Test 2 ----------------------------- \n")

    # Test 2 if a real neighbour is seen as one.
    positions_set3 = [[5, 5, 5], [5, 5, 8]]
    positions_set3 = np.asarray(positions_set3)
    nb_result3 = setup_neighbours(positions_set3, len(positions_set3), 10, 1)

    # Call the named tuple and show the positions and the distance
    print("position result 3:", nb_result3.nb_ID)
    print("distance result 3:",  nb_result3.nb_dist)
    print("---------------End Test 3 ----------------------------- \n")

if 0:
    def check_update(positions, particle_count, char_length, x):
        charges = np.random.rand(particle_count, 1)
        sigma = np.ones((particle_count, 1))
        epsilon_lj = np.ones((particle_count, 1))
        system = nbp.System(characteristic_length=char_length,
                            sigma=sigma, epsilon_lj=epsilon_lj, particle_charges=charges, positions=positions,
                            lj=True, ewald=True, use_neighbours=True)
        neighbours = nbp.Neighbours(system.info(), system.state(), system, verbose=True)
        beginning_list = neighbours.get_neighbours(x)

        # new positions
        positions = positions + 1
        system.update_state()    # what has to go in here?
        updated_list1 = neighbours.update_neighbours

        positions = positions + 1
        nbp.System(char_length, sigma, charges, positions)
        updated_list2 = neighbours.update_neighbours

        return beginning_list, updated_list1, updated_list2


    positions_set = [[0, 0, 0], [1, 1.5, 1.2], [2, 2, 2]]
    positions_set = np.asarray(positions_set)
    nb_update = check_update(positions_set, len(positions_set), 10, 1, 1)

    print("position result beginning:", nb_update[0].nb_ID)
    print("distance result beginning:",  nb_update[0].nb_dist)
    print("\n #----Next step: \n")
    print("position result updated:", nb_update[1])
    print("distance result updated:",  nb_update[1])
    print("\n #----Next step: \n")
    print("position result updated:", nb_update[2])
    print("distance result updated:",  nb_update[2])

    print("---------------End Test 4 ----------------------------- \n")
