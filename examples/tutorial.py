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
np.random.seed(314)
x  = np.linspace(0, 10, 1000)
p0 = [3, -2.4, 0.5]
y  = quad(p0, x)

uncert = np.sqrt(np.abs(y))
error  = np.random.normal(0, uncert)
data   = y + error
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Define the modeling function as a callable:
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

# Parameter prior probability distributions:
prior    = np.array([ 0.0, 0.0, 0.0])
priorlow = np.array([ 0.0, 0.0, 0.0])
priorup  = np.array([ 0.0, 0.0, 0.0])

# Parameter names:
pnames   = ['y0', 'alpha', 'beta']
texnames = [r'$y_{0}$', r'$\alpha$', r'$\beta$']

# Sampler algorithm, choose from: 'snooker', 'demc' or 'mrw'.
sampler = 'snooker'

# MCMC setup:
nsamples =  1e5
burnin   = 1000
nchains  =   14
ncpu     =    7
thinning =    1

# MCMC initial draw, choose from: 'normal' or 'uniform'
kickoff = 'normal'
# DEMC snooker pre-MCMC sample size:
hsize   = 10

# Optimization before MCMC, choose from: 'lm' or 'trf':
leastsq    = 'lm'
chisqscale = False

# MCMC Convergence:
grtest  = True
grbreak = 1.01
grnmin  = 0.5

# Logging:
log = 'MCMC_tutorial.log'

# File outputs:
savefile = 'MCMC_tutorial.npz'
plots    = True
rms      = True

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

# Run the MCMC:
mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
     indparams=indparams, pmin=pmin, pmax=pmax, pstep=pstep,
     pnames=pnames, texnames=texnames,
     prior=prior, priorlow=priorlow, priorup=priorup,
     sampler=sampler, nsamples=nsamples,  nchains=nchains,
     ncpu=ncpu, burnin=burnin, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     grtest=grtest, grbreak=grbreak, grnmin=grnmin,
     hsize=hsize, kickoff=kickoff,
     wlike=wlike, log=log,
     plots=plots, savefile=savefile, rms=rms)
