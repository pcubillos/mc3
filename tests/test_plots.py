# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import numpy as np
import mc3.plots as mp
import mc3.stats as ms


nsamples = 1000
posts = [
    np.array([
        np.random.normal(0, 1.0, nsamples)
        for _ in range(nposts)
    ]).T
    for nposts in [1,2,3,13]
]
# Relative tolerance:
rtol = 1e-6

def test_subplotter():
    rect = [0.1, 0.1, 0.9, 0.9]
    margin = 0.1
    ipan = 1
    nx = 2
    axis = mp.subplotter(rect, margin, ipan, nx)


@pytest.mark.parametrize('post', posts)
def test_trace(post):
    axes = mp.trace(post)
    # TBD: make output a Posterior object


@pytest.mark.parametrize('post', posts)
def test_pairwise(post):
    axes = mp.pairwise(post)
    # TBD: make output a Posterior object


@pytest.mark.parametrize('post', posts)
def test_histogram(post):
    axes = mp.histogram(post)
    # TBD: make output a Posterior object


def test_rms():
    data = np.random.normal(0, 1.0, nsamples)
    rms, lo, hi, stderr, binsz = ms.time_avg(data)
    ax = mp.rms(binsz, rms, stderr, lo, hi)


def test_modelfit():
    indparams = np.linspace(0.0, 1.0, nsamples)
    data = np.random.normal(1.00, 1.0, nsamples)
    uncert = np.random.normal(1.0, 0.1, nsamples)
    model = np.tile(1.0, nsamples)
    axes = mp.modelfit(data, uncert, indparams, model)


def test_alphatize_string():
    color = 'red'
    alpha = 0.5
    acol = mp.alphatize(color, alpha)

    expected_color = np.array([1.0, 0.5, 0.5])
    np.testing.assert_allclose(acol, expected_color, rtol)


def test_alphatize_rgb():
    color = (1.0, 0.0, 0.0)
    alpha = 0.5
    acol = mp.alphatize(color, alpha)

    expected_color = np.array([1.0, 0.5, 0.5])
    np.testing.assert_allclose(acol, expected_color, rtol)


def test_alphatize_rgba():
    # 'Original' alpha is pretty much ignored:
    color = (1.0, 0.0, 0.0, 0.5)
    alpha = 0.5
    acol = mp.alphatize(color, alpha)

    expected_color = np.array([1.0, 0.5, 0.5])
    np.testing.assert_allclose(acol, expected_color, rtol)


def test_alphatize_background():
    color1 = 'red'
    color2 = 'blue'
    alpha = 0.5
    acol = mp.alphatize(color1, alpha, color2)

    expected_color = np.array([0.5, 0.0, 0.5])
    np.testing.assert_allclose(acol, expected_color, rtol)


def test_alphatize_iterable():
    # Input a list of colors:
    acols = mp.alphatize(['r', 'b'], alpha=0.8)
    expected_color0 = np.array([1.0, 0.2, 0.2])
    expected_color1 = np.array([0.2, 0.2, 1.0])
    np.testing.assert_allclose(acols[0], expected_color0, rtol)
    np.testing.assert_allclose(acols[1], expected_color1, rtol)


def test_Theme():
    color = 'xkcd:blue'
    theme = mp.Theme(color)

    expected_light_color = np.array([0.25882353, 0.44705882, 0.90588235])
    expected_dark_color = np.array([0.00588235, 0.13137255, 0.4372549 ])
    expected_bad = expected_under = np.array([1., 1., 1., 1.])
    expected_first = np.array([0.85176471, 0.88941176, 0.98117647])
    expected_last = np.array([0.00588235, 0.13137255, 0.4372549 ])

    assert theme.color == color
    np.testing.assert_allclose(theme.light_color, expected_light_color, rtol)
    np.testing.assert_allclose(theme.dark_color, expected_dark_color, rtol)

    colormap = theme.colormap
    assert colormap.N == 256
    try:
        bad = colormap.get_bad()
        under = colormap.get_under()
    except AttributeError:
        # Python3.6 compatibility:
        bad = colormap._rgba_bad
        under = colormap._rgba_under
    np.testing.assert_allclose(bad, expected_bad, rtol)
    np.testing.assert_allclose(under, expected_under, rtol)
    np.testing.assert_allclose(colormap.colors[0], expected_first, rtol)
    np.testing.assert_allclose(colormap.colors[-1], expected_last, rtol)

