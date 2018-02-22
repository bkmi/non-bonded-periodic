import nbp
import numpy as np

data = nbp.Parser('sodium-chloride-example.npz').parse()

system = nbp.System(data['ch_length'], data['sigma'], data['epsilon'], data['charges'], data['pos'],
                    reci_cutoff=10, lj=True, ewald=True, use_neighbours=True)
op_sys = system.optimize()
simu_sys = op_sys.simulate(10000, 300)
