# Computational Sciences WS17/18 project: Non Bonded Periodic

A Python package for non-bonded interactions in periodic systems.

The method used to simulate the system is Markov Chain Montecarlo (MCMC). Before the simulation starts, the system is optimized until its point of minimum energy. Then the simulation starts. As the system is simulated through MCMC, it does not evolve in time, but just samples from a probability distribution.
It is possible to simulate the system at a certain temparature.

The energy is calculated using either Lennard-Jones Potential, Ewald Summation or both.
There's the option to use neighbourlist techniques to cut the calculations.


## Code structure


<p align="center">
  <img src="https://i.imgur.com/HInt9o3.png" width="900"/>
</p>

The structure of the code follows roughly this UML diagram.
The code is divided in two subgroups:
  * the acting part (on the left), that optimize and simulate the system
  * the data part (on the right), that contains all the information, both static and dynamic, about the system
Additional parts are to support, e.g. the misc and the neighbours class


## Operating instructions

These are the steps to follow to simulate a system:

```python
import nbp

sys = nbp.System(characteristic_length, sigma, epsilon_lj, particle_charges, positions, 
                 lj=True, ewald=True, use_neighbours=True)
optimized_sys = sys.optimize()
simulated_sys = optimized_sys.simulate(states, temp)
```

#### Loading the parameters from a file

A file can be used to store the starting parameters of the system. It must contain a dictionary with keys:

> ['parameters', 'positions', 'types', 'box', 'readme']

To load the parameters into the System, the Parser contained into nbp must be used, for example:

```python
import nbp.parser

data = Parser("file.npz").parse()

characteristic_length = data['ch-length']
sigma = data['sigma'][:, None]
epsilon = data['epsilon'][:, None]
charge = data['charge][:, None]
positions = data['pos']
```

Then the procedure follows as stated before.
If no file is passed, the file sodium-chloride-example.npz, contained in this repo, is used.

---

### Authors 

Theresa Kiszler  
Ludovica Lombardi  
Benjamin Miller  
Alexey Shestakov  
Chris Weiss
