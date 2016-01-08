# This script shows how to run MCMC from an interactive python sesion.

# Preamble
# --------
# To correctly execute this script, one needs to set the correct paths
# to the source code.  The paths are given as if the Python session
# runs from the MCcubed/examples/tutorial01/ folder of the repository.

# Alternatively, edit the paths from this script to adjust to your
# working directory.


# Import the necessary modules:
import sys
import numpy as np

# Import the modules from the MCcubed package:
sys.path.append("../../src")
import mccubed as mc3
import mcutils as mu
sys.path.append("./../models/")
from quadratic import quad


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0, 10, 100)          # Independent variable of the model
p0 = 3, -2.4, 0.5                     # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

mu.savebin([data, uncert], 'data.npz')
# indparams contains additional arguments of func (if necessary). Each
# additional argument is an item in the indparams tuple:
mu.savebin([x],      'indp.npz')
# Set the arguments to the file names:
data      = 'data.npz'
indparams = 'indp.npz'


# MCMC algorithm:
walk    = 'demc'  # Choose between: {'demc' or 'mrw'}


# Define the modeling function as a callable:
sys.path.append("./../models/")
from quadratic import quad
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
nchains  =  10   # Number of parallel chains
burnin   = 300   # Number of burned-in samples per chain
thinning =   1   # Thinning factor for outputs

# Optimization:
leastsq    = True   # Least-squares minimization prior to the MCMC
chisqscale = False  # Scale the data uncertainties such red.chisq = 1

# MCMC Convergence:
grtest = True
grexit = False  # TBI

# File outputs:
log       = 'MCMC.log'         # Save the MCMC screen outputs to file
savefile  = 'MCMC_sample.npy'  # Save the MCMC parameters sample to file
savemodel = 'MCMC_models.npy'  # Save the MCMC evaluated models to file
plots     = True               # Generate best-fit, trace, and posterior plots

# Correlated-noise assessment:
wlike = False   # Use Carter & Winn's Wavelet-likelihood method
rms   = False   # Compute the time-averaging test and plot


# Run the MCMC:
posterior, Zchain, bestp = mc3.mcmc(data=data,
        func=func,  indparams=indparams,
        params=params,
        walk=walk, nsamples=nsamples,  nchains=nchains,
        burnin=burnin, thinning=thinning,
        leastsq=leastsq, chisqscale=chisqscale,
        hsize=1, kickoff='normal',
        grtest=grtest, wlike=wlike, log=log,
        plots=plots,  savefile=savefile, savemodel=savemodel, resume=False)
