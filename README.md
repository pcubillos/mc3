## Multi-core Markov-chain Monte Carlo (MC<sup>3</sup>)
>A python implementation of the Markov-chain Monte Carlo algorithm.

### Table of Contents:
* [Team Members](#team-members)
* [Code Description](#code-description)
* [Examples](#examples)
* [Installation and System Requirements](#installation-and-system-requirements)
* [Coming up](#coming-up)
* [Further Reading](#further-reading)
* [Be Kind](#be-kind)
* [License](#license)


### Team Members:
* [Patricio Cubillos](https://github.com/pcubillos/) (author) <pcubillos@fulbrightmail.org>
* Madison Stemm
* Joe Harrington

### Code Description:
**TL;DR:** You provide the data and the model function, I'll give you the parameter posteriors.

MC<sup>3</sup> provides a set of routines to sample the posterior probability distributions for the model-fitting parameters.  To do so it uses Bayesian Inference through a Markov-chain Monte Carlo algorithm following, either, Differential-Evolution (recomended) or Metropolis Random Walk. It handles Bayesian priors, Gelman-Rubin convergence test, or shared parameters (p<sub>j</sub>=p<sub>i</sub>) over the MCMC iterations.  You can run MC<sup>3</sup> interactively through the python interpreter or from the terminal command line.

**Modules summary** (project's [source](src/) code):
* mcmc.py
> Core module that implements the MCMC algorithm.

* mccubed.py
> Wrapper of mcmc.py that provides support for MPI multiprocessing and use of configuration files.

* func.py
> Subroutine of mccubed.py that handles the model function in parallel under MPI.

* mcutils.py
> Utility functions used in the project's code.

* mcplots.py
> A set of functions to plot parameter trace curves, pairwise posterior dostributions, and marginalized posterior histograms.

The following sequence diagram (UML 2.0) details the interaction of the code modules under MPI:
<dl >
  <img src="doc/READMEplots/MCcubed_sequence_diagram.png" width="600">
</dl>

### Examples:
The [examples](examples/) folder contains two working and detailed examples to run MC<sup>3</sup> from an interactive python sesion and from the shell.  But, here is a quick demo anyways:


```python
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("./src/")
import mccubed as mc3
import mcplots as mp

# Get function to model (and sample):
sys.path.append("./examples/example01/")
from quadratic import quad

# Create a synthetic dataset:
x = np.linspace(0, 10, 100) # Independent model variable
p0 = 3, -2.4, 0.5 # True-underlying model parameters
y = quad(p0, x) # Noiseless model
uncert = np.sqrt(np.abs(y)) # Data points uncertainty
error = np.random.normal(0, uncert) # Noise for the data
data = y + error # Noisy data set


# To see a quick description of the MCMC arguments, run:
help(mcmc.mcmc)

# Fit the quad polynomial coefficients:
params = np.array([ 20.0, -2.0, 0.1]) # Initial guess of fitting params.

# Run the MCMC:
allp, bp = mc3.mcmc(data, uncert, func=quad, indparams=[x],
                    params=params, numit=3e4, burnin=100)

# That's it, now let's see the results:
y0 = quad(params, x) # Initial guess values
y1 = quad(bp, x) # MCMC best fitting values

plt.figure(10)
plt.clf()
plt.plot(x, y, "-k", label='true')
plt.errorbar(x, data, yerr=uncert, fmt=".b", label='data')
plt.plot(x, y0, "-g", label='Initial guess')
plt.plot(x, y1, "-r", label='MCMC best fit')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("quad(x)")
plt.suptitle("quadratic fit")
plt.savefig("quad_fit.png")

# The module mcplots provides helpful plotting functions:
# Plot trace plot:
parname = ["constant", "linear", "quadratic"]
mp.trace(allp, title="Fitting-parameter Trace Plots", parname=parname,
         savefile="quad_trace.png")

# Plot pairwise posteriors:
mp.pairwise(allp, title="Pairwise posteriors", parname=parname,
         savefile="quad_pairwise.png")

# Plot marginal posterior histograms:
mp.histogram(allp, title="Marginal posterior histograms", parname=parname,
         savefile="quad_hist.png")
```
<dl >
  <img src="doc/READMEplots/quad_fit.png"   width="400">
  <img src="doc/READMEplots/quad_trace.png" width="400">
</dl>

<dl>
  <img src="doc/READMEplots/quad_pairwise.png" width="400">
  <img src="doc/READMEplots/quad_hist.png"     width="400">
</dl>

### Installation and System Requirements:
MC<sup>3</sup> is a pure python code and doesn't need an installation.
Download the latest MC<sup>3</sup> stable version from the [Releases](https://github.com/pcubillos/MCcubed/releases) page, and start using it.  

**System Requirements:**
- The basic built-in Python libraries (os, sys, warnings, argparse, ConfigParser, subprocess, timeit, time, traceback, textwrap)
- numpy
- matplotlib
- mpi4py  (only required for multi-core runs)

### Coming Up:
Soon to be added to the repo:
- Wavelet-based Likelihood.
- Time averaging plots (RMS vs binsize).
- Levenberg-Marquardt minimization previous to MCMC.

### Further Reading:
Differential-evolution Markov chain algorithm:
[ter Braak 2006: A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution](http://dx.doi.org/10.1007/s11222-006-8769-1).

Wavelet Likelihood: [Carter & Winn 2009: Parameter Estimation from Time-series Data with Correlated Errors](http://adsabs.harvard.edu/abs/2009ApJ...704...51C).

### Be Kind:
Please reference this paper if you found this module useful for your research:  
  [Cubillos et al. 2014: A Review on the Correlated Noise Analyses Applied to Exoplanet Light Curves](https://github.com/pcubillos/demc/), in preparation.  
Thanks!

### License:
Copyright (c) 2014 Patricio Cubillos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
