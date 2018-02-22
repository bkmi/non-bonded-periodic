import nbp
import numpy as np

data = nbp.Parser('sodium-chloride-example.npz').parse()

for key, val in data.items():
    if isinstance(val, np.ndarray):
        print(key, val.shape)
        print('flipped ', val[:,None].shape)
    elif isinstance(val, dict):
        for sk, sv in val.items():
            print(sk, sv)

# units epsilon0 = 55.3e-4 eV/A
# 1 kJ/mol = 1.0364x10-2 eV

system = nbp.System(data['ch_length'],
                    data['sigma'][:,None],
                    data['epsilon'][:,None] * 1.0364e-2,
                    data['charge'][:,None],
                    data['pos'],
                    lj=True, ewald=True, use_neighbours=False,
                    epsilon0=55.3e-4)
op_sys = system.optimize()
simu_sys = op_sys.simulate(1000, 200)
