## Multi-processor Markov-chain Monte Carlo (M-cube?)
>A python implementation of the Markov-chain Monte Carlo algorithm.

### Table of Contents:
* [Team Members](#team-members)
* [Code Description](#code-description)
* [Examples](#examples)
* [Installation and System Requirements](#installation-and-system-requirements)
* [Further Reading](#further-reading)
* [License](#license)
 

### Team Members:
* [Patricio Cubillos](https://github.com/pcubillos/) (author) <pcubillos@fulbrightmail.org>
* Madison Stemm
* Joe Harrington

### Code Description:
M-cube provides a set of routines to sample the posterior probability distributions for the model-fitting parameters.  To do so it uses Bayesian Inference through a Markov-chain Monte Carlo algorithm following, either, Differential-Evolution (recomended) or Metropolis Random Walk. It handles Bayesian priors, Gelman-Rubin convergence test.

**TL;DR:** You provide the data and the model function, I'll give you the parameter posteriors.


**Modules summary** (project's [source](src/) code):
* mcmc.py
> Core module that implements MCMC.

* mpmc.py
> Wrapper of mcmc.py that provides the MPI multiprocess support.

* func.py
> Subroutine of mpmc.py to handles the model function in parallel under MPI.

* mcutils.py
> Utility functions used in the project's code.

* mcplots.py
> A set of functions to plot parameter trace curves, pairwise posterior dostributions, and marginalized posterior histograms.

The following sequence diagram (UML 2.0) details the interaction of the code modules under MPI:
![MPMC_sequence_diagram.pdf](doc/MPMC_sequence_diagram.png)

### Examples:
The [examples](examples/) folder contains two working examples to test M-cube: (1) from an interactive python session and (2) from the shell. 

Bash code:
```bash
$ gcc my_program.c `pkg-config --cflags --libs gumbo`
```

### Installation and System Requirements:
M-cube is a pure python code and doesn't need an installation, simply download the code and start using it.

**System Requirements:**
- The basic built-in Python libraries (os, sys, warnings, argparse, ConfigParser, subprocess, timeit, time)
- numpy
- matplotlib
- mpi4py

### Further Reading:
The differential-evolution Markov chain algorithm is further detailed in
[Braak 2006: A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution](http://dx.doi.org/10.1007/s11222-006-8769-1)


### License:
Copyright (c) 2014 Patricio Cubillos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
