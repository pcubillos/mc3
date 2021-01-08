# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import os
import sys
import pytest

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


np.random.seed(12)
# Create a synthetic dataset:
x = np.linspace(0, 10, 100)
p0 = [4.5, -2.4, 0.5]
y = quad(p0, x)
uncert = np.sqrt(np.abs(y))
error = np.random.normal(0, uncert)
data = y + error

p1 = [4.5, 4.5, 0.5]
y1 = quad(p1, x)
uncert1 = np.sqrt(np.abs(y1))
data1 = y1 + np.random.normal(0, uncert1)

# Fit the quad polynomial coefficients:
params   = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
pmin     = np.array([ 0.0, -5.0, -1.0])
pmax     = np.array([10.0,  5.0,  1.0])
pstep    = np.array([0.03, 0.03, 0.05])
pnames   = ["constant", "linear", "quadratic"]
texnames = ["$\\alpha$", "$\\log(\\beta)$", "quadratic"]
sampler = 'dynesty'


def test_dynesty_minimal():
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        indparams=[x], pmin=pmin, pmax=pmax,
        pstep=pstep, sampler=sampler, maxiter=5000)
    # No error? that's a pass.
    assert output is not None


def test_dynesty_ncpu():
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        indparams=[x], pmin=pmin, pmax=pmax, ncpu=8,
        pstep=pstep, sampler=sampler, maxiter=5000)
    assert output is not None


def test_dynesty_shared():
    output = mc3.sample(data1, uncert1, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=[0.03, -1, 0.05], maxiter=5000)
    assert output is not None
    assert output['bestp'][1] == output['bestp'][0]


def test_dynesty_fixed():
    pars = np.copy(params)
    pars[0] = p0[0]
    output = mc3.sample(data, uncert, func=quad, params=np.copy(pars),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=[0, 0.03, 0.05], maxiter=5000)
    assert output is not None
    assert len(output['bestp']) == len(params)
    assert output['bestp'][0] == pars[0]
    assert output['CRlo'][0] == 0
    assert output['CRhi'][0] == 0
    assert output['stdp'][0] == 0


@pytest.mark.parametrize('leastsq', ['lm', 'trf'])
def test_dynesty_optimize(capsys, leastsq):
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=pstep, maxiter=5000, 
        leastsq=leastsq)
    captured = capsys.readouterr()
    assert output is not None
    assert "Least-squares best-fitting parameters:" in captured.out
    np.testing.assert_allclose(output['bestp'],
        np.array([4.28263253, -2.40781859, 0.49534411]), rtol=1e-7)


def test_dynesty_priors_gauss():
    prior    = np.array([ 4.5,  0.0,   0.0])
    priorlow = np.array([ 0.1,  0.0,   0.0])
    priorup  = np.array([ 0.1,  0.0,   0.0])
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=pstep, maxiter=5000,
        prior=prior, priorlow=priorlow, priorup=priorup)
    assert output is not None
    assert -2*output['best_log_post'] > output['best_chisq']
    assert np.all(-2*output['log_post'] > output['chisq'])


def test_dynesty_log(capsys, tmp_path):
    os.chdir(str(tmp_path))
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=pstep, maxiter=5000,
        log='NS.log')
    captured = capsys.readouterr()
    assert output is not None
    assert "NS.log" in captured.out
    assert "NS.log" in os.listdir(".")


def test_dynesty_savefile(capsys, tmp_path):
    os.chdir(str(tmp_path))
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pmin=pmin, pmax=pmax,
        pstep=pstep, maxiter=5000,
        savefile='NS.npz')
    captured = capsys.readouterr()
    assert output is not None
    assert 'dynesty_sampler' in output
    assert "NS.npz" in captured.out
    assert "NS.npz" in os.listdir(".")


# Trigger errors:
def test_dynesty_pmin_error(capsys):
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pstep=pstep, pmax=pmax)
    captured = capsys.readouterr()
    assert output is None
    assert "Parameter space must be constrained by pmin and pmax." \
        in captured.out


def test_dynesty_pmax_error(capsys):
    output = mc3.sample(data, uncert, func=quad, params=np.copy(params),
        sampler=sampler, indparams=[x], pstep=pstep, pmin=pmin)
    captured = capsys.readouterr()
    assert output is None
    assert "Parameter space must be constrained by pmin and pmax." \
        in captured.out
