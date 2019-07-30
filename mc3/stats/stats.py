# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'bin_array',
    'residuals',
    'chisq',
    'dwt_chisq',
    'cred_region',
    'ppf_uniform',
    'ppf_gaussian',
    'dwt_daub4',
    ]

import sys

import numpy as np
import scipy.stats       as ss
import scipy.interpolate as si

from .. import utils as mu
sys.path.append(mu.ROOT + 'mc3/lib/')
import _binarray as ba
import _chisq    as cs
import _dwt      as dwt


def bin_array(data, binsize, uncert=None):
    """
    Compute the binned weighted mean and standard deviation of an array
    using 1/uncert**2 as weights.
    Eq. (4.31) of Data Reduction and Error Analysis for the Physical
    Sciences by Bevington & Robinson).

    Parameters
    ----------
    data: 1D ndarray
        A time-series dataset.
    binsize: Integer
        Number of data points per bin.
    uncert: 1D ndarray
        Uncertainties of data (if None, assume that all data points have
        same uncertainty).

    Returns
    -------
    bindata: 1D ndarray
        Mean-weighted binned data.
    binunc: 1D ndarray
        Standard deviation of the binned data points (returned only if
        uncert is not None).

    Notes
    -----
    If the last bin does not contain binsize elements, it will be
    trnucated from the output.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ndata = 12
    >>> data   = np.array([0,1,2, 3,3,3, 3,3,4])
    >>> uncert = np.array([3,1,1, 1,2,3, 2,2,4])
    >>> binsize = 3
    >>> # Binning, no weights:
    >>> bindata = ms.bin_array(data, binsize)
    >>> print(bindata)
    [1.         3.         3.33333333]
    >>> # Binning using uncertainties as weights:
    >>> bindata, binstd = ms.bin_array(data, binsize, uncert)
    >>> print(bindata)
    [1.42105263 3.         3.11111111]
    >>> print(binstd)
    [0.6882472  0.85714286 1.33333333]

    """
    if uncert is None:
        return ba.binarray(np.array(data, dtype=np.double), int(binsize))
    return ba.binarray(np.array(data, dtype=np.double), int(binsize),
                       np.array(uncert, dtype=np.double))


def residuals(model, data, uncert,
        params=None, priors=None, priorlow=None, priorup=None):
    """
    Calculate the residuals between a dataset and a model

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    errors: 1D ndarray
        Data uncertainties.
    params: 1D float ndarray
        Model parameters.
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Prior lower uncertainty.
    priorup: 1D ndarray
        Prior upper uncertainty.

    Returns
    -------
    residuals: 1D ndarray
        Residuals array.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data   = np.array([1.1, 1.2, 0.9, 1.0])
    >>> model  = np.array([1.0, 1.0, 1.0, 1.0])
    >>> uncert = np.array([0.1, 0.1, 0.1, 0.1])
    >>> res = ms.residuals(model, data, uncert)
    print(res)
    [-1. -2.  1.  0.]
    >>> # Now, say this is a two-parameter model, with a uniform and
    >>> # a Gaussian prior, respectively:
    >>> params = np.array([2.5, 5.5])
    >>> priors = np.array([2.0, 5.0])
    >>> plow   = np.array([0.0, 1.0])
    >>> pup    = np.array([0.0, 1.0])
    >>> res = ms.residuals(model, data, uncert, params, priors, plow, pup)
    >>> print(res)
    [-1.  -2.   1.   0.   0.5]
    """
    if params is None or priors is None or priorlow is None or priorup is None:
        return cs.residuals(model, data, uncert)

    iprior = (priorlow > 0) & (priorup > 0)
    dprior = (params - priors)[iprior]
    return cs.residuals(model, data, uncert, dprior,
        priorlow[iprior], priorup[iprior])


def chisq(model, data, uncert,
          params=None, priors=None, priorlow=None, priorup=None):
    """
    Calculate chi-squared of a model fit to a data set:
        chisq = sum{data points} ((data[i] -model[i])/error[i])**2.0

    If params, priors, priorlow, and priorup are not None, calculate:
        chisq = sum{data points} ((data[i] -model[i])/error[i])**2.0
              + sum{priors} ((params[j]-prior[j])/prioruncert[j])**2.0
    Which is not chi-squared, but is the quantity to optimize when a
    parameter has a Gaussian prior (equivalent to maximize the Bayesian
    posterior probability).

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    uncert: 1D ndarray
        Data uncertainties.
    params: 1D float ndarray
        Model parameters.
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        The chi-squared value.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data   = np.array([1.1, 1.2, 0.9, 1.0])
    >>> model  = np.array([1.0, 1.0, 1.0, 1.0])
    >>> uncert = np.array([0.1, 0.1, 0.1, 0.1])
    >>> chisq  = ms.chisq(model, data, uncert)
    print(chisq)
    6.0
    >>> # Now, say this is a two-parameter model, with a uniform and
    >>> # a Gaussian prior, respectively:
    >>> params = np.array([2.5, 5.5])
    >>> priors = np.array([2.0, 5.0])
    >>> plow   = np.array([0.0, 1.0])
    >>> pup    = np.array([0.0, 1.0])
    >>> chisq = ms.chisq(model, data, uncert, params, priors, plow, pup)
    >>> print(chisq)
    6.25
    """
    if params is None or priors is None or priorlow is None or priorup is None:
        return cs.chisq(model, data, uncert)

    iprior = (priorlow > 0) & (priorup > 0)
    dprior = (params - priors)[iprior]
    return cs.chisq(model, data, uncert, dprior,
                    priorlow[iprior], priorup[iprior])


def dwt_chisq(model, data, params, priors=None, priorlow=None, priorup=None):
    """
    Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared)
    based on Carter & Winn (2009), ApJ 704, 51.

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    params: 1D float ndarray
        Model parameters (including the tree noise parameters: gamma,
        sigma_r, sigma_w; which must be the last three elements in params).
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        Wavelet-based (pseudo) chi-squared.

    Notes
    -----
    - If the residuals array size is not of the form 2**N, the routine
    zero-padds the array until this condition is satisfied.
    - The current code only supports gamma=1.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np

    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
    >>> model = np.ones(8)
    >>> params = np.array([1.0, 0.1, 0.1])
    >>> chisq = ms.chisq(model, data, params)
    >>> print(chisq)
    1693.22308882
    """
    if priors is None or priorlow is None or priorup is None:
        return dwt.chisq(params, model, data)

    iprior = (priorlow > 0) & (priorup > 0)
    dprior = (params - priors)[iprior]
    return dwt.chisq(params, model, data, dprior, priorlow, priorup)


def cred_region(posterior=None, quantile=0.6827, pdf=None, xpdf=None,
    # Deprecated: Remove by 2020-07-01
    percentile=None):
    """
    Compute the highest-posterior-density credible region for a
    posterior distribution.

    Parameters
    ----------
    posterior: 1D float ndarray
        A posterior distribution.
    quantile: Float
        The HPD quantile considered for the credible region.
        A value in the range: (0, 1).
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    percentile: Float
        Deprecated. Use quantile instead.

    Returns
    -------
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    HPDmin: Float
        The minimum density in the percentile-HPD region.

    Example
    -------
    >>> import numpy as np
    >>> import mc3.stats as ms
    >>> # Test for a Normal distribution:
    >>> npoints = 100000
    >>> posterior = np.random.normal(0, 1.0, npoints)
    >>> pdf, xpdf, HPDmin = ms.cred_region(posterior)
    >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

    >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
    >>> pdf, xpdf, HPDmin = ms.cred_region(pdf=pdf, xpdf=xpdf, quantile=0.9545)
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
    """
    if percentile is not None:
        with mu.Log() as log:
            log.warning('percentile is deprecated, use quantile instead.')
        quantile = percentile

    if pdf is None and xpdf is None:
        # Thin if posterior has too many samples (> 120k):
        thinning = np.amax([1, int(np.size(posterior)/120000)])
        # Compute the posterior's PDF:
        kernel = ss.gaussian_kde(posterior[::thinning])
        # Remove outliers:
        mean = np.mean(posterior)
        std  = np.std(posterior)
        k = 6
        lo = np.amax([mean-k*std, np.amin(posterior)])
        hi = np.amin([mean+k*std, np.amax(posterior)])
        # Use a Gaussian kernel density estimate to trace the PDF:
        x  = np.linspace(lo, hi, 100)
        # Interpolate-resample over finer grid (because kernel.evaluate
        #  is expensive):
        f    = si.interp1d(x, kernel.evaluate(x))
        xpdf = np.linspace(lo, hi, 3000)
        pdf  = f(xpdf)

    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])
    # Indices of the highest posterior density:
    iHPD = np.where(cdf >= quantile*cdf[-1])[0][0]
    # Minimum density in the HPD region:
    HPDmin = np.amin(pdf[ip][0:iHPD])
    return pdf, xpdf, HPDmin


def ppf_uniform(pmin, pmax):
    """
    Percent-point function (PPF) for a uniform function between
    pmin and pmax.  Also known as inverse CDF or quantile function.

    Parameters
    ----------
    pmin: Float
        Lower boundary of the uniform function.
    pmax: Float
        Upper boundary of the uniform function.

    Returns
    -------
    ppf: Callable
        The uniform's PPF.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ppf_u = ms.ppf_uniform(-10.0, 10.0)
    >>> # The domain of the output function is [0,1]:
    >>> ppf_u(0.0), ppf_u(0.5), ppf_u(1.0)
    (-10.0, 0.0, 10.0)
    >>> # Also works for np.array inputs:
    >>> print(ppf_u(np.array([0.0, 0.5, 1.0])))
    array([-10.,   0.,  10.])
    """
    def ppf(u):
        return (pmax-pmin)*u + pmin
    return ppf


def ppf_gaussian(loc, lo, up):
    """
    Percent-point function (PPF) for a two-sided Gaussian function
    Also known as inverse CDF or quantile function.

    Parameters
    ----------
    loc: Float
        Center of the Gaussian function.
    lo: Float
        Left-sided standard deviation (for values x < loc).
    up: Float
        Right-sided standard deviation (for values x > loc).

    Returns
    -------
    ppf: Callable
        The Gaussian's PPF.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ppf_g = ms.ppf_gaussian(0.0, 1.0, 1.0)
    >>> # The domain of the output function is [0,1]:
    >>> ppf_g(1e-10), ppf_g(0.5), ppf_g(1.0-1e-10)
    (-6.361340902404056, 0.0, 6.361340889697422)
    >>> # Also works for np.array inputs:
    >>> print(ppf_g(np.array([1e-10, 0.5, 1-1e-10])))
    [-6.3613409   0.          6.36134089]
    """
    def ppf(u):
        if np.isscalar(u) and u < lo/(lo+up):
            return ss.norm.ppf(0.5*u*(lo+up)/lo, scale=lo, loc=loc)
        elif np.isscalar(u):
            return ss.norm.ppf(1-0.5*(1-u)*(lo+up)/up, scale=up, loc=loc)
        # else:
        icdf = np.empty_like(u)
        left = u < lo/(lo+up)
        icdf[ left] = ss.norm.ppf(0.5*u[left]*(lo+up)/lo, scale=lo, loc=loc)
        icdf[~left] = ss.norm.ppf(1-0.5*(1-u[~left])*(lo+up)/up, scale=up,
            loc=loc)
        return icdf
    return ppf


def dwt_daub4(array, inverse=False):
    """
    1D discrete wavelet transform using the Daubechies 4-parameter wavelet

    Parameters
    ----------
    array: 1D ndarray
        Data array to which to apply the DWT.
    inverse: bool
        If False, calculate the DWT,
        If True, calculate the inverse DWT.

    Notes
    -----
    The input vector must have length 2**M with M an integer, otherwise
    the output will zero-padded to the next size of the form 2**M.

    Examples
    --------
    >>> import numpy as np
    >>> improt matplotlib.pyplot as plt
    >>> import mc3.stats as ms

    >>> # Calculate the inverse DWT for a unit vector:
    >>> nx = 1024
    >>> e4 = np.zeros(nx)
    >>> e4[4] = 1.0
    >>> ie4 = ms.dwt_daub4(e4, True)
    >>> # Plot the inverse DWT:
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(np.arange(nx), ie4)
    """
    isign = -1 if inverse else 1
    return dwt.daub4(np.array(array), isign)
