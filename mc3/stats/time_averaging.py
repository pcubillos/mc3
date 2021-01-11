# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['time_avg']

import sys

import numpy as np

from .. import utils as mu
sys.path.append(mu.ROOT + 'mc3/lib')
import timeavg as ta


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
    """
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    
    if maxbins is None:
        maxbins = len(data) // 2

    return ta.binrms(data, int(maxbins), int(binstep))
