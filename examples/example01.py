# This example script runs mcmc.py for a quadratic function from an
# interactive python sesion.

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../src")
import mcmc as mcmc

# Get a funtion to model/fit/sample.
from quadratic import quad

# Create a synthetic dataset:
x  = np.linspace(0, 10, 100)  # Independent model variable
p0 = 3, -2.4, 0.5             # True-underlying model parameters
y  = quad(p0, x)              # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

# Set the MCMC arguments:
indparams = x

# We will fit the polynomial coefficients:
params   = np.array([ 20.0,  -2.0,   0.1])
pmin     = np.array([-10.0, -20.0, -10.0])
pmax     = np.array([ 40.0,  20.0,  10.0])
stepsize = np.array([  1.0,   0.5,   0.1])
prior    = np.array([  0.0,   0.0,   0.0]) 
priorup  = np.array([  0.0,   0.0,   0.0])
priorlow = np.array([  0.0,   0.0,   0.0])
parscale = np.array([  0.1,   0.1,   0.1])

# MCMC setup:
numit   = 3e4
nchains = 10
walk    = 'demc'  # or 'mrw'
grtest  = True
burnin  = 100

# Define function for MCMC:
# As callable:
func = quad
# Or by function name and module name:
#     func = ("quad", "quadratic")
# If the module is out of the python-path scope, set it as the 3rd parameter:
#     func = ("quad", "quadratic", "path/to/file")

# Run the MCMC:
allp, bp = mcmc.mcmc(data, uncert, func, indparams,
            params, pmin, pmax, stepsize,
            numit=numit, nchains=nchains, walk=walk, grtest=grtest,
            burnin=burnin)

# Print out the results:
print("The true parameters were: " + str(p0) +
      "\nThe fit  parameters are:  " + str(np.mean(allp, 1)) + 
      "\nWith uncertainties:       " + str(np.std(allp,  1)))

# Evaluate and plot:
y0 = quad(params, x)  # Initial guess values
y1 = quad(bp,     x)  # MCMC best fitting values

plt.figure(10)
plt.clf()
plt.plot(x, y, "-k",   label='true')
plt.errorbar(x, data, yerr=uncert, fmt=".b", label='data')
plt.plot(x, y0, "-g",  label='guess')
plt.plot(x, y1, "-r",  label='MCMC')
plt.legend(loc="best")

# Plot trace plot:
plt.figure(20)
plt.clf()
plt.subplot(311)
plt.title("Fitting-parameter Trace Plots")
plt.plot(allp[0])
plt.ylabel("P0")
plt.subplot(312)
plt.plot(allp[1])
plt.ylabel("P1")
plt.subplot(313)
plt.plot(allp[2])
plt.ylabel("P2")
plt.xlabel("Iterations")

# Plot pairwise posteriors:
plt.figure(30)
plt.clf()
plt.subplot(221)
plt.title("Fitting-parameter Pairwise Posteriors")
plt.plot(allp[0], allp[1], ",")
plt.ylabel("P1")

plt.subplot(223)
plt.plot(allp[0], allp[2], ",")
plt.ylabel("P2")
plt.xlabel("P0")

plt.subplot(224)
plt.plot(allp[1], allp[2], ",")
plt.xlabel("P1")

