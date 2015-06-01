# This script shows how to run MCMC from an interactive python sesion, 
# for a quadratic function.

# This script assumes that your current folder is /examples/example01/

# ::::: Running from an interactive python session ::::::::::::::::::

# The module mccubed.py (multi-core MCMC) wraps around mcmc.py to enable
# mutiprocessor capacity (using MPI), and use of configuration files.

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../src")
import mccubed as mc3
import mcplots as mp
import mcutils as mu

# Get function to model/sample.
sys.path.append("../")
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
help(mc3.mcmc)  # Displays the MCMC function docstring.

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
# item in the indparams list:
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
savefile  = 'output_ex1.npy'
savemodel = 'output_model.npy'

# Run the MCMC:
allp, bp = mc3.mcmc(data, uncert, func, indparams,
            params, pmin, pmax, stepsize,
            numit=numit, nchains=nchains, walk=walk, grtest=grtest,
            burnin=burnin, plots=plots, savefile=savefile, savemodel=savemodel)


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


# ::::: Multi-core Markov-chain Monte Carlo :::::::::::::::::::::::::
# A multi-process MCMC will use one CPU for each MCMC-chain
# to calculate the model for the set of parameters in that chain.
# To use MPI set the mpi argument to True, and run mc3.mcmc as usual:
mpi=True
allp, bp = mc3.mcmc(data, uncert, func, indparams,
            params, pmin, pmax, stepsize,
            numit=numit, nchains=nchains, walk=walk, grtest=grtest,
            burnin=burnin, plots=plots, savefile=savefile, mpi=mpi)


# ::::::: Arguments as files ::::::::::::::::::::::::::::::::::::::::
# As said in the help description, the data, uncert, indparams, params, 
# pmin, pmax, stepsize, prior, priorlow, and priorup arrays can be
# read from a text file.  In this case, set the argument to be the file name.

# Each line in the 'indparams' file must contain one element of the indparams
# list, the values separated by (one or more) empty spaces.
# The other files must contain one array value per line (i.e., column-wise). 

# Furthermore, the 'data' file can also contain the uncert array (as a second
# column, values separated by a empty space).
# Likewise, the 'params' file can contain the pmin, pmax, stepsize, prior,
# priorlow, and priorup arrays (as many or as few, provided that they are
# written in columns in that precise order).

# The mcutils module provides the function 'writedata' to easily make these
# files in the required format, for example:
mu.writebin([data, uncert],                  'data_ex01.dat')
mu.writebin(indparams,                       'indp_ex01.dat')
mu.writedata([params, pmin, pmax, stepsize], 'pars_ex01.dat')
# Check them out.  These files can contain empty or comment lines without
# interfering with the routine.

# Set the arguments to the file names:
data      = 'data_ex01.dat'
params    = 'pars_ex01.dat'
indparams = 'indp_ex01.dat'
# Run MCMC:
allp, bp = mc3.mcmc(data=data, func=func, indparams=indparams,
                    params=params,
                    numit=numit, nchains=nchains, walk=walk, grtest=grtest,
                    burnin=burnin, plots=plots, savefile=savefile)


# ::::: Configuration files :::::::::::::::::::::::::::::::::::::::::
# MC3 Also supports the use of a configuration file to set up the mcmc
# arguments.  Use the cfile argument of mc3.mcmc to specify a configuration
# file:
cfile = "my_config_file.cfg"
# Check out example02 to see a configuration file example.
# In case of conflict, when an argument is specified twice, the
# inline-command value will override the configuration-file value.
