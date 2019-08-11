import sys
import numpy as np
import mc3


def quad(p, x):
    """
    Quadratic polynomial function.

    Parameters
        p: Polynomial constant, linear, and quadratic coefficients.
        x: Array of dependent variables where to evaluate the polynomial.
    Returns
        y: Polinomial evaluated at x:  y(x) = p0 + p1*x + p2*x^2
    """
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Preamble (create a synthetic dataset, in a real scenario you would
# get your dataset from your own data analysis pipeline):
np.random.seed(3)
x  = np.linspace(0, 10, 100)
p0 = [3, -2.4, 0.5]
y  = quad(p0, x)
uncert = np.sqrt(np.abs(y))
data   = y + np.random.normal(0, uncert)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Modeling function:
func = quad
# List of additional arguments of func (if necessary):
indparams = [x]

# Array of initial-guess values of fitting parameters:
params = np.array([ 10.0, -2.0, 0.1])
# Lower and upper boundaries for the MCMC exploration:
pmin = np.array([-10.0, -20.0, -10.0])
pmax = np.array([ 40.0,  20.0,  10.0])
# Parameters' stepping behavior:
pstep = np.array([1.0, 0.5, 0.1])

# Two-sided Gaussian prior on first parameter, uniform priors on rest:
prior    = np.array([3.5, 0.0, 0.0])
priorlow = np.array([0.1, 0.0, 0.0])
priorup  = np.array([0.3, 0.0, 0.0])

# Parameter names:
pnames   = ['y0', 'alpha', 'beta']
texnames = [r'$y_{0}$', r'$\alpha$', r'$\beta$']

# Sampler algorithm, choose from: 'snooker', 'demc', 'mrw', or 'dynesty'.
sampler = 'dynesty'

# Optimization before MCMC, choose from: 'lm' or 'trf':
leastsq    = 'lm'
chisqscale = False
# NS setup:
ncpu     = 7
thinning = 1

# Logging:
log = 'NS_tutorial.log'

# File outputs:
savefile = 'NS_tutorial.npz'
plots    = True
rms      = True

# Run the MCMC:
mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
     indparams=indparams, pmin=pmin, pmax=pmax, pstep=pstep,
     pnames=pnames, texnames=texnames,
     prior=prior, priorlow=priorlow, priorup=priorup,
     sampler=sampler, ncpu=ncpu, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     plots=plots, rms=rms, log=log, savefile=savefile)
