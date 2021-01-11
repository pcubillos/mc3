# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

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
params   = np.array([10.0, -2.0, 0.1])


def test_fit_minimal():
    output = mc3.fit(data, uncert, quad, np.copy(params), indparams=[x])
    np.testing.assert_allclose(output['best_log_post'], -54.43381306220858)
    np.testing.assert_equal(-2*output['best_log_post'], output['best_chisq'])
    np.testing.assert_allclose(output['bestp'],
        np.array([4.28263253, -2.40781859, 0.49534411]), rtol=1e-7)


def test_fit_trf():
    output = mc3.fit(data, uncert, quad, np.copy(params), indparams=[x],
        leastsq='trf')
    np.testing.assert_allclose(output['best_log_post'], -54.43381306220856)
    np.testing.assert_allclose(output['bestp'],
        np.array([4.28263252, -2.40781858, 0.49534411]), rtol=1e-7)


def test_fit_shared():
    output = mc3.fit(data1, uncert1, quad, np.copy(params), indparams=[x],
        pstep=[1.0, -1, 1.0])
    assert output['bestp'][1] == output['bestp'][0]
    np.testing.assert_allclose(output['best_log_post'], -51.037667264657)
    np.testing.assert_allclose(output['bestp'],
        np.array([4.58657213, 4.58657213, 0.43347714]), rtol=1e-7)


def test_fit_fixed():
    pars = np.copy(params)
    pars[0] = p0[0]
    output = mc3.fit(data, uncert, quad, pars, indparams=[x],
        pstep=[0.0, 1.0, 1.0])
    assert output['bestp'][0] == pars[0]
    np.testing.assert_allclose(output['best_log_post'], -54.507722717665466)
    np.testing.assert_allclose(output['bestp'],
        np.array([4.5, -2.51456999, 0.50570154]), rtol=1e-7)


def test_fit_bounds():
    output = mc3.fit(data, uncert, quad, [4.5, -2.5, 0.5], indparams=[x],
        pmin=[4.4, -3.0, 0.4], pmax=[5.0, -2.0, 0.6], leastsq='trf')
    np.testing.assert_allclose(output['best_log_post'], -54.45536109795812)
    np.testing.assert_allclose(output['bestp'],
        np.array([4.4, -2.46545897, 0.5009366]), rtol=1e-7)


def test_fit_priors():
    prior    = np.array([ 4.5,  0.0,   0.0])
    priorlow = np.array([ 0.1,  0.0,   0.0])
    priorup  = np.array([ 0.1,  0.0,   0.0])
    output = mc3.fit(data, uncert, quad, np.copy(params), indparams=[x],
        prior=prior, priorlow=priorlow, priorup=priorup)
    np.testing.assert_allclose(output['best_log_post'], -54.50548056991611)
    # First parameter is closer to 4.5 than without a prior:
    np.testing.assert_allclose(output['bestp'],
        np.array([4.49340587, -2.51133157, 0.50538734]), rtol=1e-7)


def test_fit_leastsq_error(capsys):
    with pytest.raises(SystemExit):
        output = mc3.fit(data, uncert, quad, np.copy(params), indparams=[x],
            leastsq='invalid')
        captured = capsys.readouterr()
        assert "Invalid 'leastsq' input (invalid). Must select from " \
               "['lm', 'trf']." in captured.out
