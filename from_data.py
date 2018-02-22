import nbp
import numpy as np

data = nbp.Parser('sodium-chloride-example.npz').parse()

for key, val in data.items():
    if isinstance(val, np.ndarray):
        print(key, val.shape)
        print('flipped ', val[:,None].shape)

system = nbp.System(data['ch_length'],
                    data['sigma'][:,None],
                    data['epsilon'][:,None],
                    data['charge'][:,None],
                    data['pos'],
                    reci_cutoff=10, lj=True, ewald=True, use_neighbours=False)
op_sys = system.optimize()
simu_sys = op_sys.simulate(1000, 200)
