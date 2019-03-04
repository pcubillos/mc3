# This script shows how to run MCMC from an interactive python sesion
# using input files.

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

# Import the modules from the MCcubed package:
sys.path.append("../MCcubed/")
import MCcubed as mc3

sys.path.append("../MCcubed/examples/models/")
from quadratic import quad


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0, 10, 100)          # Independent variable of the model
p0 = 3, -2.4, 0.5                     # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

mu = mc3.utils
mu.savebin([data, uncert], 'data.npz')
# indparams contains additional arguments of func (if necessary). Each
# additional argument is an item in the indparams tuple:
mu.savebin([x],      'indp.npz')
# Set the arguments to the file names:
data      = 'data.npz'
indparams = 'indp.npz'


# MCMC algorithm:
walk    = 'snooker'  # Choose between: {'demc' or 'mrw'}


# Define the modeling function as a callable:
func = quad  # The first argument of func() must be the fitting parameters


# Array of initial-guess values of fitting parameters:
pars     = np.array([ 20.0,  -2.0,   0.1])
# Lower and upper boundaries for the MCMC exploration:
pmin     = np.array([-10.0, -20.0, -10.0])
pmax     = np.array([ 40.0,  20.0,  10.0])
# Parameters stepsize:
stepsize = np.array([  1.0,   0.5,   0.1])
# Parameter prior probability distributions:
prior    = np.array([ 0.0,  0.0,   0.0]) # The prior value
priorlow = np.array([ 0.0,  0.0,   0.0])
priorup  = np.array([ 0.0,  0.0,   0.0])

# The mcutils module provides the function 'saveascii' to easily make these
# files in the required format, for example:
mu.saveascii([pars, pmin, pmax, stepsize, prior, priorlow, priorup],
             'parameters.dat')
params = 'parameters.dat'


# MCMC sample setup:
nsamples = 1e5   # Number of MCMC samples to compute
nchains  =   7   # Number of parallel chains
burnin   = 300   # Number of burned-in samples per chain
thinning =   1   # Thinning factor for outputs

# Initial sample:
kickoff = 'normal' # Choose between: 'normal' or  'uniform'
hsize = 10         # Number of initial samples per chain

# Optimization:
leastsq    = True   # Least-squares minimization prior to the MCMC
lm         = True   # Choose Levenberg-Marquardt (True) or TRF algorithm (False)
chisqscale = False  # Scale the data uncertainties such red.chisq = 1

# MCMC Convergence:
grtest  = True
grbreak = 1.001
grnmin = 0.6

# File outputs:
log       = 'MCMC.log'         # Save the MCMC screen outputs to file
savefile  = 'MCMC_sample.npz'  # Save the MCMC parameters sample to file
plots     = True               # Generate best-fit, trace, and posterior plots

# Correlated-noise assessment:
wlike = False   # Use Carter & Winn's Wavelet-likelihood method
rms   = True    # Compute the time-averaging test and plot


# Run the MCMC:
bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data=data,
        func=func,  indparams=indparams,
        params=params,
        walk=walk, nsamples=nsamples,  nchains=nchains,
        burnin=burnin, thinning=thinning,
        leastsq=leastsq, lm=lm, chisqscale=chisqscale,
        hsize=hsize, kickoff=kickoff,
        grtest=grtest, grbreak=grbreak, grnmin=grnmin,
        wlike=wlike, log=log,
        plots=plots,  savefile=savefile, rms=rms)
