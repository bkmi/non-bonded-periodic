import nbp
import numpy as np


def make_system():
    box_length = 10
    sigma = 0.5
    particle_count = 50
    random_positions = box_length * np.random.rand(particle_count, 3)
    random_charges = np.random.rand(particle_count, 1)
    system = nbp.System(box_length, sigma, random_charges, random_positions)
    return system


def test_no_self_in_neighbours():
    system = make_system()

    self_in_neighbours = []
    for particle, position in enumerate(system.state().positions()):
        result = system.state().neighbours().get_neighbours(position)
        if particle in result.nb_pos:
            self_in_neighbours.append(True)
        else:
            self_in_neighbours.append(False)

    assert any(self_in_neighbours) is False


def test_communicativity():
    system = make_system()

    for particle, position in enumerate(system.state().positions()):
        result = system.state().neighbours().get_neighbours(position)
        nb_pos = result.nb_pos
        nb_dist = result.nb_dist
        print('Neighbours of particle {0}: {1}'.format(particle, nb_pos))
        print('Distances from particle {0}: {1}'.format(particle, nb_dist))
        for neighbour_index, neighbour in enumerate(nb_pos):
            neighbour_result = system.state().neighbours().get_neighbours(
                system.state().positions()[neighbour])
            neighbour_neighbours = neighbour_result.nb_pos
            neighbour_dist = neighbour_result.nb_dist
            print('Neighbours of particle {0}: {1}'.format(neighbour, neighbour_neighbours))
            print('Distances from particle {0}: {1}'.format(neighbour, neighbour_dist))

            if particle not in neighbour_neighbours:
                assert particle in neighbour_neighbours

            if neighbour_dist[neighbour_neighbours.index(particle)] != nb_dist[neighbour_index]:
                assert neighbour_dist[neighbour_neighbours.index(particle)] == nb_dist[neighbour_index]
        assert True


if 0:
    def setup_neighbours(positions, particle_number, char_length, sigma, x):
        charges = np.random.random_sample((particle_number, 1))
        system = nbp.System(char_length, sigma, charges, positions)
        neighbours = nbp.Neighbours(system.info(), system.state(), system, verbose=True)
        neighbours_list = neighbours.get_neighbours(positions[x])

        return neighbours_list


    positions_set1 = 4 * np.random.random_sample((100, 3))
    nb_result = setup_neighbours(positions_set1, len(positions_set1), 5, 0.5, 8)

    # Call the named tuple and show the positions and the distance
    print("position result 1:", nb_result.nb_pos)
    print("distance result 1:",  nb_result.nb_dist)
    print("------------------End Test 1 ----------------------------- \n")

    # Test if a real neighbour is seen as one.
    positions_set2 = [[0, 0, 0], [1, 1.5, 1.2], [10, 10, 10]]
    positions_set2 = np.asarray(positions_set2)
    nb_result2 = setup_neighbours(positions_set2, len(positions_set2), 10, 1, 0)

    # Call the named tuple and show the positions and the distance
    print("position result 2:", nb_result2.nb_pos)
    print("distance result 2:",  nb_result2.nb_dist)
    print("--------------End Test 2 ----------------------------- \n")

    # Test 2 if a real neighbour is seen as one.
    positions_set3 = [[5, 5, 5], [5, 5, 8]]
    positions_set3 = np.asarray(positions_set3)
    nb_result3 = setup_neighbours(positions_set3, len(positions_set3), 10, 1, 0)

    # Call the named tuple and show the positions and the distance
    print("position result 3:", nb_result3.nb_pos)
    print("distance result 3:",  nb_result3.nb_dist)
    print("---------------End Test 3 ----------------------------- \n")

if 0:
    def check_update(positions, particle_number, char_length, sigma, x):
        charges = np.random.random_sample((particle_number, 1))
        system = nbp.System(char_length, sigma, charges, positions)
        neighbours = nbp.Neighbours(system.info(), system.state(), system, verbose=True)
        beginning_list = neighbours.get_neighbours(positions[x])

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

    print("position result beginning:", nb_update[0].nb_pos)
    print("distance result beginning:",  nb_update[0].nb_dist)
    print("\n #----Next step: \n")
    print("position result updated:", nb_update[1])
    print("distance result updated:",  nb_update[1])
    print("\n #----Next step: \n")
    print("position result updated:", nb_update[2])
    print("distance result updated:",  nb_update[2])

    print("---------------End Test 4 ----------------------------- \n")
