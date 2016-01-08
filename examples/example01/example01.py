# This script shows how to run MCMC from an interactive python sesion.

# Preamble
# --------
# To correctly execute this script, one needs to set the correct paths
# to the source code.  The paths are given as if the Python session
# runs from the MCcubed/examples/example01/ folder of the repository.

# Alternatively, edit the paths from this script to adjust to your
# working directory.


# Import the necessary modules:
import sys, os
import numpy as np

sys.path.append("../../src/")
import mcmc as mcmc
import chain as ch
sys.path.append("../models/")
from quadratic import quad

# Create a synthetic dataset:
x  = np.linspace(0, 10, 5000)         # Independent model variable
p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

indparams = [x]

# Define as callable:
func = quad


walk = "demc"   # Choose between: {'demc' or 'mrw'}
nchains = 10    # Number of parallel chains
nsamples = 1e5  # Number of MCMC samples to compute
thinning = 2    # Thinning factor for outputs
burnin = 900    # Number of burned-in samples per chain


params = [4, -2, 0.2]
stepsize = [0.03, 0.03, 0.005]
# Parameter prior probability distributions:
pmin = [-10.0, -10.0, -10.0]


reload(ch)
reload(mcmc)
reload(mp)
# Run the MCMC:
Z, Zchain, bp = mcmc.mcmc(data, uncert, func,  indparams=indparams,
   params=params,  pmin=pmin, pmax=None, stepsize=stepsize,
   prior=None,   priorlow=None,    priorup=None,
   nsamples=nsamples,  nchains=nchains,  walk=walk, wlike=False,
   leastsq=True, chisqscale=True, grtest=True,  burnin=burnin,
   thinning=thinning, hsize=1, kickoff='normal',
   plots=True,  savefile='theoutput', savemodel=None, resume=False)
