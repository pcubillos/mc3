# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'cred_region',
    'ppf_uniform',
    'ppf_gaussian',
    ]

import numpy as np
import scipy.stats       as ss
import scipy.interpolate as si

from .. import utils as mu


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
    >>> import MCcubed.stats as ms
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

