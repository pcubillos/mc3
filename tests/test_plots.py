# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import numpy as np
import mc3.plots as mp
import mc3.stats as ms


nsamples = 1000
posts = [np.array([np.random.normal(0, 1.0, nsamples)
                   for _ in range(nposts)]).T
         for nposts in [1,2,3,13]]


def test_subplotter():
    rect = [0.1, 0.1, 0.9, 0.9]
    margin = 0.1
    ipan = 1
    nx = 2
    axis = mp.subplotter(rect, margin, ipan, nx)


@pytest.mark.parametrize('post', posts)
def test_trace(post):
    axes = mp.trace(post)


@pytest.mark.parametrize('post', posts)
def test_pairwise(post):
    axes, colorbar = mp.pairwise(post)


@pytest.mark.parametrize('post', posts)
def test_histogram(post):
    axes = mp.histogram(post)


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

