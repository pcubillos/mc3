# Copyright (c) 2015-2026 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'time_avg',
]

import numpy as np

from .stats import bin_array
from .. import utils as mu


def inv_gamma(M: int, s: float, ds: float):
    """
    Compute credible region for inverse-gamma–like posterior
    of Equation 33 of Cubillos et al. (2017)

    Parameters
    ----------
    M: int
        Degrees-of-freedom-like parameter.
    s: float
        Central scale parameter.
    ds: float
        Standard deviation scale.

    Returns
    -------
    low, high: tuple of floats
        Lower and upper bounds relative to s.
    """
    n = 10000
    xmax = s + 50.0 * ds
    xmin = s - 4.0 * ds
    if xmin < 0.01 * s:
        xmin = 0.01 * s

    x = np.linspace(xmin, xmax, n)
    posterior = x**-M * np.exp(-M * s**2.0 / (2.0*x**2.0))
    posterior /= np.sum(posterior)

    # Find highest CDF region containing 0.683 of the PDF
    isort = np.argsort(posterior)[::-1]
    cdf = np.cumsum(posterior[isort])
    i = np.searchsorted(cdf, 0.683)
    bound1 = x[isort][i]
    if bound1 > s:
        while x[isort][i] > s:
            i += 1
    else:
        while x[isort][i] < s:
            i += 1
    bound2 = x[isort][i]
    low = s - np.amin([bound1, bound2])
    high = np.amax([bound1, bound2]) - s

    # Compute CDF until 68.3%
    #cdf1 = 0.5 * (1-0.683)
    #cdf2 = 1.0 - cdf1
    #cdf_vals = np.cumsum(posterior)
    #i_lo, i_hi = np.searchsorted(cdf_vals, (cdf1, cdf2))
    #low = s - x[i_lo]
    #high = x[i_hi] - s

    return low, high


def time_avg(data, maxbins=None, binstep=1):
    """
    Compute the binned root-mean-square and extrapolated
    Gaussian-noise RMS for a dataset.

    Parameters
    ----------
    data: 1D float ndarray
        A time-series dataset.
    maxbins: Integer
        Maximum bin size to calculate, default: len(data)/2.
    binstep: Integer
        Stepsize of binning indexing.

    Returns
    -------
    rms: 1D float ndarray
        RMS of binned data.
    rmslo: 1D float ndarray
        RMS lower uncertainties.
    rmshi: 1D float ndarray
        RMS upper uncertainties.
    stderr: 1D float ndarray
        Extrapolated RMS for Gaussian noise.
    binsz: 1D float ndarray
        Bin sizes.

    Notes
    -----
    This function uses an asymptotic approximation to obtain the
    rms uncertainties (rms_error = rms/sqrt(2M)) when the number of
    bins is M > 35.
    At smaller M, the errors become increasingly asymmetric. In this
    case the errors are numerically calculated from the posterior
    PDF of the rms (an inverse-gamma distribution).
    See Cubillos et al. (2017), AJ, 153, 3.

    >>> import mc3.stats as ms
    >>> 
    >>> # Generate mock residuals signals:
    >>> N = 1000
    >>> np.random.seed(16)
    >>> white = np.random.normal(0, 5, N)
    >>> red = np.sin(np.arange(N)/(0.1*N))*np.random.normal(1.0, 1.0, N)
    >>> 
    >>> # Plot the time-correlated residuals signal:
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(white+red, ".k")
    >>> plt.ylabel("Residuals", fontsize=14)
    >>> 
    >>> # Compute rms-vs-binsize for non-correlated and time-correlated signal
    >>> maxbins = N//5
    >>> ta_white = ms.time_avg(white, maxbins)
    >>> ta_red = ms.time_avg(white+red, maxbins)
    >>> 
    >>> # Plot rms + Gaussian standard deviation:
    >>> plt.figure(17)
    >>> plt.clf()
    >>> plt.title('Non-correlated noise')
    >>> plt.errorbar(
    >>>     ta_white[4], ta_white[0], yerr=ta_white[1:3],
    >>>     fmt="k-", ecolor='0.5', label="Data RMS",
    >>> )
    >>> plt.loglog(
    >>>     ta_white[4], ta_white[3],
    >>>     color='r', ls='-', lw=2, label="Gaussian std",
    >>> )
    >>> plt.xlim(1,200)
    >>> plt.legend(loc="upper right")
    >>> plt.xlabel("Bin size", fontsize=14)
    >>> plt.ylabel("RMS", fontsize=14)

    >>> plt.figure(16)
    >>> plt.clf()
    >>> plt.title('Time-correlated noise')
    >>> plt.errorbar(
    >>>     ta_red[4], ta_red[0], yerr=ta_red[1:3],
    >>>     fmt="k-", ecolor='0.5', label="Data RMS",
    >>> )
    >>> plt.loglog(
    >>>     ta_red[4], ta_red[3],
    >>>     color='r', ls='-', lw=2, label="Gaussian std",
    >>> )
    >>> plt.xlim(1,200)
    >>> plt.legend(loc="upper right")
    >>> plt.xlabel("Bin size", fontsize=14)
    >>> plt.ylabel("RMS", fontsize=14)
    """
    ndata = len(data)
    if maxbins is None:
        maxbins = ndata // 2

    size = (int(maxbins)-1) // int(binstep) + 1
    rms  = np.zeros(size)
    rms_lo = np.zeros(size)
    rms_hi = np.zeros(size)
    gausserr = np.zeros(size)
    binsize = np.zeros(size, int)

    # Calculate standard deviation of data
    std_data = np.std(data)

    # Flag to indicate when to recompute the IG CR
    Mtemp = -1
    for i in range(size):
        binsize[i] = 1 + i*int(binstep)
        M = ndata // binsize[i]
        bin_data = bin_array(data, binsize[i])

        # Calculate the rms
        rms[i] = np.sqrt(np.sum(bin_data**2.0)/M)
        rms_lo[i] = rms_hi[i] = rms[i]/np.sqrt(2.0*M)

        # Extrapolate Gaussian-noise rms
        gausserr[i] = std_data * np.sqrt(M/(binsize[i]*(M - 1.0)))

        # Large-bin-size regime:
        # Start computing at M=35, when RMSerror ~10%
        if M <= 35:
            if Mtemp != M:  # Compute low, high only once for each new M
                ds = gausserr[i] / np.sqrt(2.0*M)
                lo, hi = inv_gamma(M, gausserr[i], ds)
                Mtemp = M

            # Renormalize the error bars
            rms_lo[i] = lo * rms[i]/gausserr[i]
            rms_hi[i] = hi * rms[i]/gausserr[i]

    return rms, rms_lo, rms_hi, gausserr, binsize

