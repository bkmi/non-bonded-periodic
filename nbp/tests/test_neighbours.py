import nbp
import numpy as np


def test_neighbours(positions, particle_number, char_length, sigma, x):
    charges = np.random.random_sample((particle_number, 1))
    system_info = nbp.SystemInfo(char_length, sigma, charges)
    system_state = nbp.SystemState(positions)
    neighbours = nbp.Neighbours(system_info, system_state)
    neighbours_list = neighbours.get_neighbours(positions[x])

    return neighbours_list


positions_set1 = 4 * np.random.random_sample((100, 3))
nb_result = test_neighbours(positions_set1, len(positions_set1), 5, 0.5, 8)

# Call the named tuple and show the positions and the distance
print("position result 1:", nb_result.nb_pos)
print("distance result 1:",  nb_result.nb_dist)
print("------------------End Test 1 ----------------------------- \n")

k = 0
if k is 0:
    # Test if a real neighbour is seen as one.
    positions_set2 = [[0, 0, 0], [1, 1.5, 1.2], [10, 10, 10]]
    positions_set2 = np.asarray(positions_set2)
    nb_result2 = test_neighbours(positions_set2, len(positions_set2), 10, 1, 0)

    # Call the named tuple and show the positions and the distance
    print("position result 2:", nb_result2.nb_pos)
    print("distance result 2:",  nb_result2.nb_dist)
    print("--------------End Test 2 ----------------------------- \n")


## Test 2 if a real neighbour is seen as one.
positions_set3 = [[5, 5, 5], [5, 5, 8]]
positions_set3 = np.asarray(positions_set3)
nb_result3 = test_neighbours(positions_set3, len(positions_set3), 10, 1, 0)

# Call the named tuple and show the positions and the distance
print("position result 3:", nb_result3.nb_pos)
print("distance result 3:",  nb_result3.nb_dist)
print("---------------End Test 3 ----------------------------- \n")
