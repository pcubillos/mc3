# This script is an example of mcmc.py
# Fits a quadratic model to simulated data

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../src")
import mcmc as mcmc

# Define the fitting function:
def quad(p, x):
  return p[0] + p[1]*x + p[2]*x**2


# Create a dataset:
x  = np.linspace(0, 10, 100)  # Independent model variable
p0 = 3, -2.4, 0.5             # True-underlying model parameters
y  = quad(p0, x)              # Noiseless model
uncert    = np.sqrt(np.abs(y))       # Data points uncertainty
error = np.random.normal(0, uncert)  # Noise of data
data      = y + error                # Noisy data set

# Set the MCMC arguments:
indparams = x
params   = np.array([ 20.0,  -2.0,   0.1])
pmin     = np.array([-10.0, -20.0, -10.0])
pmax     = np.array([ 40.0,  20.0,  10.0])
stepsize = np.array([  1.0,   0.5,   0.1])
prior    = np.array([  0.0,   0.0,   0.0]) 
priorup  = np.array([  0.0,   0.0,   0.0])
priorlow = np.array([  0.0,   0.0,   0.0])
parscale = np.array([  0.1,   0.1,   0.1])
numit   = 3e4
nchains = 10
walk    = 'demc'
grtest  = True
burnin  = 100


# Define function wrapper:
def func(params, indparams):
  nchains, nparams = np.shape(params)
  models = []
  for c in np.arange(nchains):
    models.append(quad(params[c], indparams))
  return models


# Run the MCMC:
allp, bp = mcmc.mcmc(data, uncert, func, indparams,
            params, pmin, pmax, stepsize,
            numit=numit, nchains=nchains, walk=walk, grtest=grtest,
            burnin=burnin)

# Results:
print("The true parameters were: " + str(p0) +
      "\nThe fit  parameters are:  " + str(np.mean(allp, 1)) + 
      "\nWith uncertainties:       " + str(np.std(allp,  1)))

y0 = quad(params, x)
y1 = quad(bp,     x)

# Plot best MCMC fit:
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
plt.plot(allp[0])
plt.ylabel("P0")
plt.subplot(312)
plt.plot(allp[1])
plt.ylabel("P1")
plt.subplot(313)
plt.plot(allp[2])
plt.ylabel("P2")
plt.xlabel("Iterations")
plt.legend(loc="best")

# Plot pairwise posteriors:
plt.figure(30)
plt.clf()
plt.subplot(221)
plt.plot(allp[0], allp[1], ",")
plt.ylabel("P1")

plt.subplot(223)
plt.plot(allp[0], allp[2], ",")
plt.ylabel("P2")
plt.xlabel("P0")

plt.subplot(224)
plt.plot(allp[1], allp[2], ",")
plt.xlabel("P1")

