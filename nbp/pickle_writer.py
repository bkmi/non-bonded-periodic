import nbp
import pickle


def create_system(temperature, ewald_flag=True, lj_flag=True, neigh_flag=True):
    data = nbp.Parser('C:/Users/ludov/Documents/Uni/Computational Sciences/git/non-bonded-periodic/sodium-chloride-example.npz').parse()

    sys0 = nbp.System(data['ch_length'],
                      data['sigma'][:, None],
                      data['epsilon'][:, None] * 1.0364e-2,
                      data['charge'][:, None],
                      data['pos'],
                      lj=lj_flag, ewald=ewald_flag, use_neighbours=neigh_flag,
                      epsilon0=55.3e-4)
    sys0.optimize(max_steps=500, cov=sys0.info().cutoff()/2**8, num_particles=0.05)
    sys0.simulate(100, temperature)

    return sys0


systems = []
for temp in range(1, 300, 30):
    systems.append({'system': create_system(temp, ewald_flag=False, neigh_flag=False), 'temperature': temp})

for temp in range(360, 1020, 60):
    systems.append({'system': create_system(temp, ewald_flag=False), 'temperature': temp})

file = open('C:/Users/ludov/Documents/Uni/Computational Sciences/git/non-bonded-periodic/data/'
            'pickle_temperatures_no_neighbours.npy', 'bw')
pickle.dump(systems, file)
file.close()

systems = []
for temp in range(1, 300, 30):
    systems.append({'system': create_system(temp, ewald_flag=False), 'temperature': temp})

for temp in range(360, 1020, 60):
    systems.append({'system': create_system(temp, ewald_flag=False), 'temperature': temp})

file = open('C:/Users/ludov/Documents/Uni/Computational Sciences/git/non-bonded-periodic/data/'
            'pickle_temperatures.npy', 'bw')
pickle.dump(systems, file)
file.close()


