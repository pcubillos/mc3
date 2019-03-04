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

# Import the modules from the MCcubed package:
sys.path.append("../MCcubed/")
import MCcubed as mc3
sys.path.append("../MCcubed/examples/models/")
from quadratic import quad


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0, 10, 1000)         # Independent variable of the model
p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set


# Define the modeling function as a callable:
# The first argument of func() must be the fitting parameters
sys.path.append("../MCcubed/examples/models/")
from quadratic import quad
func = quad

# A three-elements tuple indicates the function name, the module 
# name (without the '.py' extension), and the path to the module.
func = ("quad", "quadratic", "../MCcubed/examples/models/")

# Alternatively, if the module is already within the scope of the
# python-path, the user can set func with a two-elements tuple:
sys.path.append("../MCcubed/examples/models/")
func = ("quad", "quadratic")


# indparams contains additional arguments of func (if necessary). Each
# additional argument is an item in the indparams tuple:
indparams = [x]


# Array of initial-guess values of fitting parameters:
params   = np.array([ 10.0,  -2.0,   0.1])
# In this case, the polynomial coefficients of the quadratic function.

# Lower and upper boundaries for the MCMC exploration:
pmin     = np.array([-10.0, -20.0, -10.0])
pmax     = np.array([ 40.0,  20.0,  10.0])

# stepsize determines the standard deviation of the proposal Gaussian function:
# For Metropolis Random Walk, the Gaussian function draws the parameter
# proposals for each iteration.
# For Differential Evolution, the Gaussian function draws the
# starting values of the chains about the initial-guess values.
stepsize = np.array([  1.0,   0.5,   0.1])

# Parameter prior probability distributions:
# priorlow defines whether to use uniform non-informative (priorlow = 0.0),
# Jeffreys non-informative (priorlow < 0.0), or Gaussian (priorlow > 0.0)
# priors, prior and priorup are irrelevant if priorlow <= 0 (for a given
# parameter).
prior    = np.array([ 0.0,  0.0,   0.0])
priorlow = np.array([ 0.0,  0.0,   0.0])
priorup  = np.array([ 0.0,  0.0,   0.0])


# MCMC algorithm, choose between: 'snooker', 'demc' or 'mrw':
walk    = 'snooker'

# MCMC sample setup:
nsamples =  1e5   # Number of MCMC samples to compute
nchains  =    7   # Number of parallel chains
nproc    =    7   # Number of CPUs to use for chains (default: nchains)
burnin   = 1000   # Number of burned-in samples per chain
thinning =    1   # Thinning factor for outputs

# Initial sample:
kickoff = 'normal' # Choose between: 'normal' or  'uniform'
hsize = 10         # Number of initial samples per chain

# Optimization:
leastsq    = True   # Least-squares minimization prior to the MCMC
lm         = True   # Choose Levenberg-Marquardt (True) or TRF algorithm (False)
chisqscale = False  # Scale the data uncertainties such red.chisq = 1

# MCMC Convergence:
grtest  = True   # Calculate the GR convergence test
grbreak = 0.0    # GR threshold to stop the MCMC run
grnmin  = 0.5    # Minimum fraction or number of samples before grbreak

# File outputs:
log       = 'MCMC.log'         # Save the MCMC screen outputs to file
savefile  = 'MCMC_sample.npz'  # Save the MCMC parameters sample to file
plots     = True               # Generate best-fit, trace, and posterior plots
full_output = False            # Return the full posterior sample
chireturn = False

# Correlated-noise assessment:
wlike = False   # Use Carter & Winn's Wavelet-likelihood method
rms   = True    # Compute the time-averaging test and plot

# Fine-tuning (only edit if it's reaaaaally necessary):
fgamma   = 1.0  # Scale factor for DEMC's gamma jump.
fepsilon = 0.0  # Jump scale factor for DEMC's "e" distribution

# Run the MCMC:
bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data=data,
        uncert=uncert, func=func,  indparams=indparams,
        params=params,  pmin=pmin, pmax=pmax, stepsize=stepsize,
        prior=prior,    priorlow=priorlow,    priorup=priorup,
        walk=walk, nsamples=nsamples,  nchains=nchains,
        nproc=nproc, burnin=burnin, thinning=thinning,
        leastsq=leastsq, lm=lm, chisqscale=chisqscale,
        fgamma=fgamma, fepsilon=fepsilon,
        hsize=hsize, kickoff=kickoff,
        grtest=grtest, grbreak=grbreak, grnmin=grnmin,
        wlike=wlike, log=log,
        plots=plots,  savefile=savefile, rms=rms,
        full_output=full_output, chireturn=chireturn)
