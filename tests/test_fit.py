import pytest

import numpy as np

import MCcubed as mc3


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
x = np.linspace(0, 10, 20)
p0 = [3.0, -2.4, 0.5]
y = quad(p0, x)
uncert = np.sqrt(np.abs(y))
error = np.random.normal(0, uncert)
data = y + error

params = np.array([10.0, -2.0, 0.1])


expected_chisq = 23.432793680742382
expected_bestp = np.array([ 2.58494691, -2.34232411,  0.51421891])

expected_bestm = np.array(
    [ 2.58494691e+00,  1.49458766e+00,  6.89114233e-01,  1.68526632e-01,
     -6.71751434e-02, -1.79910943e-02,  3.16078779e-01,  9.35034478e-01,
      1.83887600e+00,  3.02760335e+00,  4.50121652e+00,  6.25971552e+00,
      8.30310034e+00,  1.06313710e+01,  1.32445275e+01,  1.61425698e+01,
      1.93254979e+01,  2.27933118e+01,  2.65460116e+01,  3.05835972e+01])


@pytest.mark.parametrize('leastsq', ['lm', 'trf'])
def test_minimal(leastsq):
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data, uncert=uncert, indparams=[x], leastsq=leastsq)
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, expected_chisq, decimal=7)
    np.testing.assert_almost_equal(bestp, expected_bestp, decimal=7)
    np.testing.assert_almost_equal(best_model, expected_bestm, decimal=7)


def test_explicit_pstep():
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data, uncert=uncert, indparams=[x],
        pstep=np.array([0.5, 0.5, 0.5]))
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, expected_chisq, decimal=7)
    np.testing.assert_almost_equal(bestp, expected_bestp, decimal=7)
    np.testing.assert_almost_equal(best_model, expected_bestm, decimal=7)


def test_fixed():
    pars = np.array([10.0, -2.4, 0.1])
    output = mc3.fit(params=pars, func=quad,
        data=data, uncert=uncert, indparams=[x],
        pstep=np.array([0.5, 0.0, 0.5]))
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, 23.44763472464093, decimal=7)
    np.testing.assert_almost_equal(bestp,
        np.array([2.68032576, -2.4, 0.52076306]), decimal=7)
    np.testing.assert_almost_equal(best_model, np.array(
        [ 2.68032576e+00,  1.56142356e+00,  7.31032752e-01,  1.89153337e-01,
         -6.42146864e-02, -2.90713174e-02,  2.94583443e-01,  9.06749596e-01,
          1.80742714e+00,  2.99661608e+00,  4.47431641e+00,  6.24052813e+00,
          8.29525124e+00,  1.06384857e+01,  1.32702316e+01,  1.61904889e+01,
          1.93992576e+01,  2.28965377e+01,  2.66823291e+01,  3.07566320e+01]),
        decimal=7)


def test_shared():
    y = quad([3.0, 3.0, 0.5], x)
    data_shared = y + error
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data_shared, uncert=uncert, indparams=[x],
        pstep=np.array([0.5, -1.0, 0.5]))
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, 23.568026730901536, decimal=7)
    assert bestp[0] == bestp[1]
    np.testing.assert_almost_equal(bestp,
        np.array([2.88751993, 2.88751993, 0.53250116]), decimal=7)
    np.testing.assert_almost_equal(best_model, np.array(
        [ 2.88751993,  4.55477451,  6.51704358,  8.77432715, 11.32662521,
         14.17393777, 17.31626482, 20.75360637, 24.48596241, 28.51333295,
         32.83571798, 37.45311751, 42.36553153, 47.57296005, 53.07540307,
         58.87286058, 64.96533258, 71.35281908, 78.03532007, 85.01283556]),
        decimal=7)


def test_bounds_lm():
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data, uncert=uncert, indparams=[x],
        pmin=[0.0, 0.0, 0.0], pmax=[40.0, 20.0, 10.0])
    chisq, bestp, best_model, lsfit = output
    # pmin/pmax has no effect on LM fits:
    np.testing.assert_almost_equal(chisq, expected_chisq, decimal=7)
    np.testing.assert_almost_equal(bestp, expected_bestp, decimal=7)
    np.testing.assert_almost_equal(best_model, expected_bestm, decimal=7)


def test_bounds_trf():
    output = mc3.fit(params=np.array([10.0, 1.0, 0.1]), func=quad,
        data=data, uncert=uncert, indparams=[x], leastsq='trf',
        pmin=[-40.0, 0.0, -10.0], pmax=[40.0, 20.0, 10.0])
    chisq, bestp, best_model, lsfit = output
    # pmin/pmax does affect TRF fits:
    np.testing.assert_almost_equal(chisq, 47.91045518323738, decimal=7)
    np.testing.assert_almost_equal(bestp,
        np.array([-1.28856370e+00, 7.82944448e-29, 2.48448985e-01]), decimal=7)
    np.testing.assert_almost_equal(best_model, np.array(
        [-1.2885637 , -1.21974127, -1.01327397, -0.6691618 , -0.18740476,
          0.43199714,  1.18904391,  2.08373555,  3.11607205,  4.28605343,
          5.59367967,  7.03895077,  8.62186675, 10.34242759, 12.2006333 ,
         14.19648387, 16.32997931, 18.60111962, 21.0099048 , 23.55633485]),
        decimal=7)


def test_priors_uniform():
    prior    = np.array([0.0, 0.0, 0.0])
    priorlow = np.array([0.0, 0.0, 0.0])
    priorup  = np.array([0.0, 0.0, 0.0])
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data, uncert=uncert, indparams=[x],
        prior=prior, priorlow=priorlow, priorup=priorup)
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, expected_chisq, decimal=7)
    np.testing.assert_almost_equal(bestp, expected_bestp, decimal=7)
    np.testing.assert_almost_equal(best_model, expected_bestm, decimal=7)


def test_priors_gauss():
    prior    = np.array([0.0, -2.4, 0.0])
    priorlow = np.array([0.0,  0.01, 0.0])
    priorup  = np.array([0.0,  0.01, 0.0])
    output = mc3.fit(params=np.copy(params), func=quad,
        data=data, uncert=uncert, indparams=[x],
        prior=prior, priorlow=priorlow, priorup=priorup)
    chisq, bestp, best_model, lsfit = output
    np.testing.assert_almost_equal(chisq, 23.447628106337433, decimal=7)
    np.testing.assert_almost_equal(bestp,
        np.array([ 2.6802832, -2.3999743,  0.5207601]), decimal=7)
    np.testing.assert_almost_equal(best_model, np.array(
        [ 2.68028323e+00,  1.56139376e+00,  7.31014061e-01,  1.89144141e-01,
         -6.42160043e-02, -2.90663741e-02,  2.94593031e-01,  9.06762211e-01,
          1.80744117e+00,  2.99662990e+00,  4.47432840e+00,  6.24053668e+00,
          8.29525474e+00,  1.06384826e+01,  1.32702202e+01,  1.61904676e+01,
          1.93992247e+01,  2.28964916e+01,  2.66822684e+01,  3.07565548e+01]),
        decimal=7)

