import sys
import os
import random

import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__) + '/..') + '/'
sys.path.append(ROOT)
import MCcubed as mc3

sys.path.append(ROOT+"examples/models/")
from quadratic import quad


np.random.seed(12)
# Create a synthetic dataset:
x = np.linspace(0, 10, 100)          # Independent model variable
p0 = [4.5, -2.4, 0.5]                # True-underlying model parameters
y = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))          # Data points uncertainty
error = np.random.normal(0, uncert)  # Noise for the data
data = y + error                     # Noisy data set

# Fit the quad polynomial coefficients:
params   = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
stepsize = np.array([0.03, 0.03, 0.05])
pnames   = ["constant", "linear", "quadratic"]
texnames = ["$\\alpha$", "$\\log(\\beta)$", "quadratic"]


def test_minimal():
    bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data, uncert,
        func=quad, indparams=[x], params=params, stepsize=stepsize,
        nsamples=1e4, burnin=100)
    # No error? that's a pass.


def test_data_error(capsys):
    MCMC = mc3.mcmc(uncert=uncert, func=quad,
        indparams=[x], params=params, stepsize=stepsize,
        nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert MCMC is None
    assert "'data' is a required argument." in captured.out


def test_uncert_error(capsys):
    MCMC = mc3.mcmc(data=data, func=quad,
        indparams=[x], params=params, stepsize=stepsize,
        nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert MCMC is None
    assert "'uncert' is a required argument." in captured.out


def test_func_error(capsys):
    MCMC = mc3.mcmc(data=data, uncert=uncert,
        indparams=[x], params=params, stepsize=stepsize,
        nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert MCMC is None
    assert "'func' must be either a callable or an iterable" in captured.out


def test_params_error(capsys):
    MCMC = mc3.mcmc(data=data, uncert=uncert, func=quad,
        indparams=[x], stepsize=stepsize,
        nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert MCMC is None
    assert "'params' is a required argument" in captured.out


def test_samples_error(capsys):
    MCMC = mc3.mcmc(data=data, uncert=uncert, func=quad,
        indparams=[x], params=params, stepsize=stepsize,
        nsamples=1e4, burnin=2000)
    captured = capsys.readouterr()
    assert MCMC is None
    assert "The number of burned-in samples (2000) is greater" in captured.out

