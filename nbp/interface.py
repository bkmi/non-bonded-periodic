import numpy as np
import matplotlib.pyplot as plt
from nbp import energy as en

r_init = []
charge = []
sampleSize = 5
iterations = 100

iX = []
eY = []


x = 0
while x <= sampleSize:
    r_init.append([np.random.uniform(0.1, 10.0), (np.random.uniform(0.1, 10.0)), (np.random.uniform(0.1, 10.0))])
    charge.append(np.random.uniform(-10.0, 10.0))
    x += 1


def random_position(sample):
    x_random = 0
    r_help = []
    while x_random <= sample:
        r_help.append([np.random.uniform(0.1, 10.0), (np.random.uniform(0.1, 10.0)), (np.random.uniform(0.1, 10.0))])
        x_random += 1
    return r_help


def optimize(r, e_vec):
    i = 0
    number = 0
    r_opt = r
    while i < iterations:
        r_test = random_position(sampleSize)
        if (en.make_energy(r_test, e_vec, sampleSize) < en.make_energy(r_opt, e_vec, sampleSize)):
            r_opt = r_test
            iX.append(number)
            number += 1
            eY.append(en.make_energy(r_test, e_vec, sampleSize))
            print(en.make_energy(r_test, e_vec, sampleSize))        #for debugging
        i += 1
    return r_opt


def simulate():
    pass


print('########')
optimize(r_init, charge)
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.plot(iX, eY, 'k')
plt.show()
print('done')
