import nbp
import numpy as np

import matplotlib.pyplot as plt

data = nbp.Parser('sodium-chloride-example.npz').parse()

for key, val in data.items():
    if isinstance(val, np.ndarray):
        print(key, val.shape)
        print('flipped ', val[:, None].shape)
    elif isinstance(val, dict):
        for sk, sv in val.items():
            print(sk, sv)

# units epsilon0 = 55.3e-4 eV/A
# 1 kJ/mol = 1.0364x10-2 eV

if 0:
    # for simulating
    system = nbp.System(data['ch_length'],
                        data['sigma'][:, None],
                        data['epsilon'][:, None] * 1.0364e-2,
                        data['charge'][:, None],
                        data['pos'],
                        lj=True, ewald=True, use_neighbours=False,
                        epsilon0=55.3e-4)
    op_sys = system.optimize()
    simu_sys = op_sys.simulate(10, 2000)
else:
    # for analysis
    traj = np.load('data/trajectory_300.npy')
    lj = np.load('data/lj_300.npy')

    system = nbp.System(data['ch_length'],
                        np.ones((traj.shape[1], 1)),
                        np.ones((traj.shape[1], 1)),
                        np.ones((traj.shape[1], 1)),
                        traj[0],
                        lj=True, ewald=False, use_neighbours=False,
                        epsilon0=1)
    for i in traj[1:50]:
        system.update_state(nbp.SystemState(i, system))

energies = []
for i in range(0, len(system.states())):
    energies.append(system.states()[i].energy())

plt.plot(np.asarray(energies))
plt.show()

print('ok')
