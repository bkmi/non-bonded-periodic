import numpy as np
import scipy.special as sci

epsilon = 1
sigma = 1
gamma = 1/(8 * np.pi * epsilon)


def make_neighborlist():
    pass


def make_short_energy(r, charge, sampleSize):   #this is just wrong but I wanted to test some energylevels
    short = 0
    for n in range(sampleSize):
        for i in range(sampleSize):
            for j in range(sampleSize):
                if i != j:
                    x = r[i][0] - r[j][0]
                    y = r[i][1] - r[j][1]
                    z = r[i][2] - r[j][2]
                    r_abs = np.sqrt(x*x + y*y + z*z)
                    short = (charge[i]*charge[j] / r_abs) * sci.erfc(r_abs / (np.sqrt(2)*sigma))

    short_energy = gamma * short
    return short_energy


def make_long_energy(r, charge):
    long_energy = 0
    return long_energy


def make_self_energy(r, charge):
    self_energy = 0
    return self_energy


def make_energy(r, charge, sampleSize):
    shortEnergy = make_short_energy(r, charge, sampleSize)
    longEnergy = make_long_energy(r, charge)
    selfEnergy = make_self_energy(r, charge)

    energy = shortEnergy + longEnergy - selfEnergy
    return energy
