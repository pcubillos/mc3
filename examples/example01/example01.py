# This script runs mcmc.py for a quadratic function from an
# interactive python sesion.

# ::::: Run this script in an interactive python session ::::::::::::
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../src")
import mcmc as mcmc
import mcplots as mp

# Get function to model/sample.
from quadratic import quad

# Create a synthetic dataset:
x  = np.linspace(0, 10, 100)  # Independent model variable
p0 = 3, -2.4, 0.5             # True-underlying model parameters
y  = quad(p0, x)              # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

# Set the MCMC arguments:
# -----------------------
# Run: 'help(mcmc.mcmc)'  to see a quick description of the MCMC arguments.

# Define the function to model the data in the MCMC: As a requirement, the
# first argument of func must be the set of fitting parameters.

# Define as callable:
func = quad
# Or by function name and module name:
#     func = ("quad", "quadratic")
# If the module is out of the python-path scope, set it as the 3rd parameter:
#     func = ("quad", "quadratic", "path/to/file")


# We will fit the quad polynomial coefficients:
params   = np.array([ 20.0,  -2.0,   0.1]) # Initial guess of fitting params.
pmin     = np.array([-10.0, -20.0, -10.0]) # Boundaries
pmax     = np.array([ 40.0,  20.0,  10.0])
# stepsize is used as the standard deviation of a Gaussian function:
# For Metropolis Random Walk, the Gaussian function draws the parameter
# proposals.  For Differential Evolution, the Gaussian function draws the
# starting values of the chains about the initial guess. Later, is used in
# the support distribution as a Gaussian with standard deviation
# of 0.01*stepsize.
stepsize = np.array([  1.0,   0.5,   0.1])

# indparams packs every additional parameter of func, each argument is an
# item in indparams:
indparams = [x]
# If func does not require additional arguments define indparams as:
# indparams=[], or simple leave it undefined in the mcmc call.

# MCMC setup:
numit   = 3e4
nchains = 10
walk    = 'demc'
grtest  = True
burnin  = 100
plots   = True
savefile = 'output_ex1.npy'

# Run the MCMC:
allp, bp = mcmc.mcmc(data, uncert, func, indparams,
            params, pmin, pmax, stepsize,
            numit=numit, nchains=nchains, walk=walk, grtest=grtest,
            burnin=burnin, plots=plots, savefile=savefile)

# Print out the results:
np.set_printoptions(precision=4)
print("The true parameters were:      " + str(p0) +
      "\nThe mean posteriors are:     " + str(np.mean(allp, 1)) +
      "\nThe best-fit parameters are: " + str(bp) + 
      "\nWith uncertainties:          " + str(np.std(allp,  1)))

# Evaluate and plot:
y0 = quad(params, x)  # Initial guess values
y1 = quad(bp,     x)  # MCMC best fitting values

plt.figure(10)
plt.clf()
plt.plot(x, y, "-k",   label='true')
plt.errorbar(x, data, yerr=uncert, fmt=".b", label='data')
plt.plot(x, y0, "-g",  label='Initial guess')
plt.plot(x, y1, "-r",  label='MCMC best fit')
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("quad(x)")

# The module mcplots provides helpful plotting functions:
# Plot trace plot:
parname = ["constant", "linear", "quadratic"]
mp.trace(allp, title="Fitting-parameter Trace Plots", parname=parname)

# Plot pairwise posteriors:
mp.pairwise(allp, title="Pairwise posteriors", parname=parname)

# Plot marginal posterior histograms:
mp.histogram(allp, title="Marginal posterior histograms", parname=parname)
