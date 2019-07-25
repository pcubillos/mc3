# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['binrms']

import sys

import numpy as np

from .. import utils as mu
from .. import stats as ms


# DEPRECATED: remove by summer 2020
def binrms(data, maxbins=None, binstep=1):
    """
    Compute the binned root-mean-square and extrapolated
    Gaussian-noise RMS for a dataset.

    This function has been deprecated.  Use mc3.stats.time_avg()
    instead.
    """
    with mu.Log() as log:
        log.warning('Deprecation warning: mc3.rednoise.binrms() moved to '
                    'mc3.stats.time_avg().')
    return ms.time_avg(data, maxbins, binstep)
