# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'ppf_uniform',
    'ppf_gaussian',
    ]

import numpy as np
import scipy.stats       as ss
import scipy.interpolate as si

from .. import utils as mu


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

