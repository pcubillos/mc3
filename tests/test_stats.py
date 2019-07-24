# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import numpy as np
import MCcubed.stats as ms


@pytest.mark.skip()
def test_bin_array():
    pass


@pytest.mark.skip()
def test_weighted_bin():
    pass


def test_cred_region():
    np.random.seed(2)
    posterior = np.random.normal(0, 1.0, 100000)
    pdf, xpdf, HPDmin = ms.cred_region(posterior)
    np.testing.assert_approx_equal(np.amin(xpdf[pdf>HPDmin]), -1.0,
        significant=3)
    np.testing.assert_approx_equal(np.amax(xpdf[pdf>HPDmin]), 1.0,
        significant=3)


@pytest.mark.parametrize('u, result',
    [(0.0, -10.0),
     (0.5,   0.0),
     (1.0,  10.0)])
def test_ppf_uniform_scalar(u, result):
    ppf_u = ms.ppf_uniform(-10.0, 10.0)
    assert ppf_u(u) == result


def test_ppf_uniform_array():
    ppf_u = ms.ppf_uniform(-10.0, 10.0)
    np.testing.assert_equal(ppf_u(np.array([0.0, 0.5, 1.0])),
        np.array([-10.0, 0.0, 10.]))


@pytest.mark.parametrize('u, result', 
    [(1e-10, -6.361340902404056), 
     (0.5,   0.0),
     (1.0-1e-10, 6.361340889697422)])
def test_ppf_gaussian_scalar(u, result):
    ppf_g = ms.ppf_gaussian(0.0, 1.0, 1.0)
    np.testing.assert_allclose(ppf_g(u), result)


def test_ppf_gaussian_array():
    ppf_g = ms.ppf_gaussian(0.0, 1.0, 1.0)
    u = np.array([1e-10, 0.5, 1-1e-10])
    result = np.array([-6.361340902404056, 0.0, 6.361340889697422])
    np.testing.assert_allclose(np.array(ppf_g(u)), result)

