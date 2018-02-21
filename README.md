# Computational Sciences WS17/18 project: Non Bonded Periodic

A Python package for non-bonded interactions in periodic systems.

The method used to simulate the system is Markov Chain Montecarlo. Before the simulation starts, the system is optimized until its point of minimum energy. Then the simulation starts. INFORMATIONS ABOUT TIME STEPS, INFORMATION ABOUT TEMPERATURE.

The energy is calculated using either Lennard Jones Potential, Ewald Summation or both.
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
TODO

---

### Authors 

Theresa Kiszler
Ludovica Lombardi
Benjamin Miller
Alexey Shestakov
Chris Weiss
