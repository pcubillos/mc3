# This script shows how to run MCMC from an interactive python sesion.

# Preamble
# --------
# To correctly execute this script, one needs to set the correct paths
# to the source code.  The paths are given as if the Python session
# runs from a 'run/' folder at the same level than the repo, as in:
#  rootdir/
#  |-- MCcubed/
#  `-- run/

# Alternatively, edit the paths from this script to adjust to your
# working directory.


# Import the necessary modules:
import sys
import numpy as np

sys.path.append("../MCcubed/")
import MCcubed as mc3

sys.path.append("../MCcubed/examples/models/")
from quadratic import quad

# Create a synthetic dataset:
x  = np.linspace(0, 10, 1000)         # Independent model variable
p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

params   = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
stepsize = np.array([0.03, 0.03, 0.05])

bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data, uncert,
    func=quad, indparams=[x], params=params, stepsize=stepsize,
    nsamples=1e5, burnin=1000)

