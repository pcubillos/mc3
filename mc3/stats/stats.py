# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'bin_array',
    'residuals',
    'chisq',
    'dwt_chisq',
    'log_prior',
    'cred_region',
    'ppf_uniform',
    'ppf_gaussian',
    'dwt_daub4',
    'Loglike',
    'Prior_transform',
    'marginal_statistics',
]

import sys

import numpy as np
import scipy.stats as ss
import scipy.interpolate as si

from .. import utils as mu
sys.path.append(mu.ROOT + 'mc3/lib/')
import _binarray as ba
import _chisq as cs
import _dwt as dwt


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
    return ba.binarray(
        np.array(data, dtype=np.double),
        int(binsize),
        np.array(uncert, dtype=np.double),
    )


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
    >>> import numpy as np
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
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
    >>> model = np.ones(8)
    >>> params = np.array([1.0, 0.1, 0.1])
    >>> chisq = ms.dwt_chisq(model, data, params)
    >>> print(chisq)
    1693.22308882
    >>> # Now, say this is a three-parameter model, with a Gaussian prior
    >>> # on the last parameter:
    >>> priors = np.array([1.0, 0.2, 0.3])
    >>> plow   = np.array([0.0, 0.0, 0.1])
    >>> pup    = np.array([0.0, 0.0, 0.1])
    >>> chisq = ms.dwt_chisq(model, data, params, priors, plow, pup)
    >>> print(chisq)
    1697.2230888243134
    """
    if len(params) < 3:
        raise ValueError('Wavelet chisq should have at least three parameters')

    if priors is None or priorlow is None or priorup is None:
        return dwt.chisq(params, model, data)

    iprior = (priorlow > 0) & (priorup > 0)
    dprior = (params - priors)[iprior]
    return dwt.chisq(
        params, model, data,
        dprior, priorlow[iprior], priorup[iprior],
    )


def log_prior(posterior, prior, priorlow, priorup, pstep):
    """
    Compute the log(prior) for a given sample (neglecting constant terms).

    This is meant to be the weight added by the prior to chi-square
    when optimizing a Bayesian posterior.  Therefore, there is a
    constant offset with respect to the true -2*log(prior) that can
    be neglected.

    Parameters
    ----------
    posterior: 1D/2D float ndarray
        A parameter sample of shape [nsamples, nfree].
    prior: 1D ndarray
        Parameters priors.  The type of prior is determined by priorlow
        and priorup:
            Gaussian: if both priorlow>0 and priorup>0
            Uniform:  else
        The free parameters in prior must correspond to those
        parameters contained in the posterior, i.e.:
        len(prior[pstep>0]) = nfree.
    priorlow: 1D ndarray
        Lower prior uncertainties.
    priorup: 1D ndarray
        Upper prior uncertainties.
    pstep: 1D ndarray
        Parameter masking determining free (pstep>0), fixed (pstep==0),
        and shared parameters.

    Returns
    -------
    logp: 1D float ndarray
        Sum of -2*log(prior):
        A uniform prior returns     logp = 0.0
        A Gaussian prior returns    logp = -0.5*(param-prior)**2/prior_uncert**2
        A log-uniform prior returns logp = log(1/param)

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np

    >>> # A posterior of three samples and two free parameters:
    >>> post = np.array([[3.0, 2.0],
    >>>                  [3.1, 1.0],
    >>>                  [3.6, 1.5]])

    >>> # Trivial case, uniform priors:
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.0, 0.0])
    >>> priorup  = np.array([0.0, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [0. 0. 0.]

    >>> # Gaussian prior on first parameter:
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.1, 0.0])
    >>> priorup  = np.array([0.1, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [25. 16. 1.]

    >>> # Posterior comes from a 3-parameter model, with second fixed:
    >>> prior    = np.array([3.5, 0.0, 0.0])
    >>> priorlow = np.array([0.1, 0.0, 0.0])
    >>> priorup  = np.array([0.1, 0.0, 0.0])
    >>> pstep    = np.array([1.0, 0.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [25. 16. 1.]

    >>> # Also works for a single 1D params array:
    >>> params   = np.array([3.0, 2.0])
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.1, 0.0])
    >>> priorup  = np.array([0.1, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(params, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    25.0
    """
    posterior = np.atleast_2d(posterior)

    ifree = np.where(pstep > 0)[0]
    nfree = len(ifree)
    dprior = posterior - prior[ifree]

    ifreeprior = np.where((priorlow[ifree]>0) & (priorup[ifree]>0))[0]
    ilogprior  = np.where(priorlow[ifree]<0)[0]

    for i in range(nfree):
        if i in ifreeprior:
            dprior[dprior[:,i]<0,i] /= priorlow[ifree][i]
            dprior[dprior[:,i]>0,i] /= priorup [ifree][i]
        elif i in ilogprior:
            dprior[:,i] = 2.0*np.log(posterior[:,i])
        else:
            dprior[:,i] = 0.0
    logp = -0.5*np.sum(dprior**2, axis=1)

    if np.size(logp) == 1:
        return logp[0]
    return logp


def cred_region(posterior=None, quantile=0.6827, pdf=None, xpdf=None):
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
    if pdf is None and xpdf is None:
        # Thin if posterior has too many samples (> 120k):
        thinning = np.amax([1, int(np.size(posterior)/120000)])
        # Compute the posterior's PDF:
        kernel = ss.gaussian_kde(posterior[::thinning])
        # Remove outliers:
        mean = np.mean(posterior)
        std = np.std(posterior)
        k = 6
        lo = np.amax([mean-k*std, np.amin(posterior)])
        hi = np.amin([mean+k*std, np.amax(posterior)])
        # Use a Gaussian kernel density estimate to trace the PDF:
        x = np.linspace(lo, hi, 100)
        # Interpolate-resample over finer grid (because kernel.evaluate
        # is expensive):
        f = si.interp1d(x, kernel.evaluate(x))
        xpdf = np.linspace(lo, hi, 3000)
        pdf = f(xpdf)

    if quantile is None:
        hpd_min = 0.0
        return pdf, xpdf, hpd_min

    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])
    # Indices of the highest posterior density:
    iHPD = np.where(cdf >= quantile*cdf[-1])[0][0]
    # Minimum density in the HPD region:
    HPDmin = np.amin(pdf[ip][0:iHPD])
    return pdf, xpdf, HPDmin


class ppf_uniform(object):
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
    >>> print(ppf_u(0.0), ppf_u(0.5), ppf_u(1.0))
    -10.0 0.0 10.0

    >>> # Also works for np.array inputs:
    >>> print(ppf_u(np.array([0.0, 0.5, 1.0])))
    array([-10.,   0.,  10.])
    """
    def __init__(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, u):
        return (self.pmax-self.pmin)*u + self.pmin


class ppf_gaussian(object):
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
    >>> # The domain of the output function is (0,1):
    >>> print(ppf_g(1e-10), ppf_g(0.5), ppf_g(1.0-1e-10))
    (-6.361340902404056, 0.0, 6.361340889697422)
    >>> # Also works for np.array inputs:
    >>> print(ppf_g(np.array([1e-10, 0.5, 1-1e-10])))
    [-6.3613409   0.          6.36134089]
    """
    def __init__(self, loc, lo, up):
        self.loc = loc
        self.lo = lo
        self.up = up

    def __call__(self, u):
        if np.isscalar(u) and u < self.lo/(self.lo+self.up):
            return ss.norm.ppf(0.5*u*(self.lo+self.up)/self.lo,
                scale=self.lo, loc=self.loc)
        elif np.isscalar(u):
            return ss.norm.ppf(1-0.5*(1-u)*(1+self.lo/self.up),
                scale=self.up, loc=self.loc)
        # else:
        icdf = np.empty_like(u)
        left = u < self.lo/(self.lo+self.up)
        icdf[ left] = ss.norm.ppf(0.5*u[left]*(1+self.up/self.lo),
            scale=self.lo, loc=self.loc)
        icdf[~left] = ss.norm.ppf(1-0.5*(1-u[~left])*(1+self.lo/self.up),
            scale=self.up, loc=self.loc)
        return icdf


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
    >>> import matplotlib.pyplot as plt
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


class Loglike(object):
    """
    Wrapper to compute log(likelihood)

    If there's any non-finite value in the model function
    (sign of an invalid parameter set), return a large-negative
    log likelihood (to reject the sample).
    """
    def __init__(self, data, uncert, func, params, indp, pstep):
        self.data = data
        self.uncert = uncert
        self.func = func
        self.params = params
        self.indp = indp
        self.pstep = pstep
        self.ifree = pstep>0
        self.ishare = np.where(pstep<0)[0]
    def __call__(self, params):
        self.params[self.ifree] = params
        for s in self.ishare:
            self.params[s] = self.params[-int(self.pstep[s])-1]
        model = self.func(self.params, *self.indp)
        log_like = -0.5 * np.sum(
            ((self.data - model) / self.uncert)**2.0
        )
        if not np.isfinite(log_like):
            log_like = -1.0e98
        return log_like


class Prior_transform(object):
    """Wrapper to compute the PPF of a set of parameters."""
    def __init__(self, prior, priorlow, priorup, pmin, pmax, pstep):
        self.ppf = []
        for p0, plo, pup, min, max, step in \
            zip(prior, priorlow, priorup, pmin, pmax, pstep):
            if step <= 0:
                continue
            if plo == 0.0 or pup == 0.0:
                self.ppf.append(ppf_uniform(min, max))
            else:
                self.ppf.append(ppf_gaussian(p0, plo, pup))
    def __call__(self, u):
        return [ppf(v) for ppf,v in zip(self.ppf, u)]


def marginal_statistics(
        posterior, statistics='med_central', quantile=0.683,
        pdf=None, xpdf=None,
    ):
    """
    Compute marginal-statistics summary (parameter estimate and
    confidence interval) for a posterior according to the given
    statistics and quantile.

    Note that this operates strictly over the 1D marginalized
    distributions for each parameter (thus the calculated marginal
    max-likelihood estimate won't necessarily match the global
    max-likelihood estimate).

    Parameters
    ----------
    posterior: 2D float array
        A posterior sample.
    statistics: String
        Which statistics to use, current options are:
        - med_central  Median estimate + central quantile CI
        - max_central  Max-likelihood (mode) + central quantile CI
        - max_like     Max-likelihood (mode) + highest-posterior-density CI
    quantile: Float
        Quantiles at which to compute the confidence interval.
    pdf: 1D irterable of 1D arrays
        Optional, the PDF for each parameter in the posterior.
    xpdf: 1D irterable of 1D arrays
        Optional, x-coordinate of the parameter PDFs.

    Returns
    -------
    values: 1D float array
        The parameter estimates.
    low_bounds: 1D float array
        The lower-boundary estimate of the parameters.
    high_bounds: 1D float array
        The upper-boundary estimate of the parameters.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np
    >>> import scipy.stats as ss

    >>> # Simulate a Gaussian vs. a skewed-Gaussian posterior:
    >>> np.random.seed(115)
    >>> nsample = 15000
    >>> posterior = np.array([
    >>>     np.random.normal(loc=5.0, scale=1.0, size=nsample),
    >>>     ss.skewnorm.rvs(a=3.0, loc=4.25, scale=1.5, size=nsample),
    >>> ]).T
    >>> nsamples, npars = np.shape(posterior)

    >>> # Median statistics (68% credible intervals):
    >>> median, lo_median, hi_median = ms.marginal_statistics(
    >>>     posterior, statistics='med_central',
    >>> )

    >>> # Maximum-likelihood statistics (68% credible intervals):
    >>> mode, lo_hpd, hi_hpd = ms.marginal_statistics(
    >>>     posterior, statistics='max_like',
    >>> )

    >>> print('      Median +/- err     |  Max_like +/- err')
    >>> for i in range(npars):
    >>>     err_lo = lo_median[i] - median[i]
    >>>     err_up = hi_median[i] - median[i]
    >>>     unc_lo = lo_hpd[i] - mode[i]
    >>>     unc_hi = hi_hpd[i] - mode[i]
    >>>     print(f'par{i+1}  {median[i]:.2f} {err_up:+.2f} {err_lo:+.2f}   '
    >>>           f'|  {mode[i]:.2f} {unc_hi:+.2f} {unc_lo:+.2f} '
    >>>     )
          Median +/- err     |  Max_like +/- err
    par1  5.00 +1.01 -1.00   |  5.01 +0.98 -1.04
    par2  5.26 +1.09 -0.83   |  5.11 +0.96 -0.91

    >>> plt.figure(1, (5,5.5))
    >>> plt.clf()
    >>> plt.subplots_adjust(0.12, 0.1, 0.95, 0.95, hspace=0.3)
    >>> for i in range(npars):
    >>>     ax = plt.subplot(npars,1,i+1)
    >>>     plt.hist(
    >>>         posterior[:,i], density=True, color='orange',
    >>>         bins=40, range=(1.5, 9.5),
    >>>     )
    >>>     plt.axvline(median[i], c='mediumblue', lw=2.0, label='Median')
    >>>     plt.axvline(lo_median[i], c='mediumblue', lw=1.0, dashes=(5,2))
    >>>     plt.axvline(hi_median[i], c='mediumblue', lw=1.0, dashes=(5,2))
    >>>     plt.axvline(mode[i], c='red', lw=2.0, label='Max likelihood')
    >>>     plt.axvline(lo_hpd[i], c='red', lw=1.0, dashes=(5,2))
    >>>     plt.axvline(hi_hpd[i], c='red', lw=1.0, dashes=(5,2))
    >>>     plt.xlabel(f'par {i+1}')
    >>>     if i == 0:
    >>>         plt.legend(loc='upper right')
    """
    nsamples, nparams = np.shape(posterior)
    values = np.tile(np.nan, nparams)
    low_bounds = np.tile(np.nan, nparams)
    high_bounds = np.tile(np.nan, nparams)

    if statistics is None:
        return values, low_bounds, high_bounds

    if pdf is None or xpdf is None:
        pdf = [None for _ in range(nparams)]
        xpdf = [None for _ in range(nparams)]
    # The parameter estimate:
    if statistics.startswith('med_'):
        values = np.median(posterior, axis=0)
    elif statistics.startswith('max_'):
        for i in range(nparams):
            pdf[i], xpdf[i], hpd_min = cred_region(
                posterior[:,i], quantile, pdf[i], xpdf[i],
            )
            pdf_max = np.argmax(pdf[i])
            values[i] = xpdf[i][pdf_max]

    # The confidence intervals:
    if quantile is None:
        return values, low_bounds, high_bounds

    if statistics.endswith('_central'):
        percentile_low = 100*0.5*(1-quantile)
        percentile_high = 100*0.5*(1+quantile)
        low_bounds = np.percentile(posterior, percentile_low, axis=0)
        high_bounds = np.percentile(posterior, percentile_high, axis=0)
    elif statistics.endswith('_like'):
        for i in range(nparams):
            pdf[i], xpdf[i], hpd_min = cred_region(
                posterior[:,i], quantile, pdf[i], xpdf[i],
            )
            low_bounds[i] = np.amin(xpdf[i][pdf[i]>hpd_min])
            high_bounds[i] = np.amax(xpdf[i][pdf[i]>hpd_min])

    return values, low_bounds, high_bounds

