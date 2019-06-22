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
import numpy as np

import MCcubed as mc3


def quad(p, x):
    """
    Quadratic polynomial function.

    Parameters
        p: Polynomial constant, linear, and quadratic coefficients.
        x: Array of dependent variables where to evaluate the polynomial.
    Returns
        y: Polinomial evaluated at x:  y = p0 + p1*x + p2*x^2
    """
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y


# Create a synthetic dataset:
x  = np.linspace(0, 10, 1000)         # Independent model variable
p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

params = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
pstep = np.array([0.03, 0.03, 0.05])

bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data, uncert,
    func=quad, indparams=[x], params=params, pstep=pstep,
    nsamples=1e5, burnin=1000)

