# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from .sampler_driver import *
from .fit_driver import *
from . import plots
from . import utils
from . import stats
from . import rednoise
from .VERSION import __version__


# Remove by 01.08.2020:
@utils.ignore_system_exit
def mcmc(*args, **kwargs):
    """This function has been deprecated. Use mc3.sample() instead."""
    with utils.Log() as log:
        log.error('mcmc() function is deprecated. Use mc3.sample() instead.')
    return


__all__ = (
    sampler_driver.__all__
  + fit_driver.__all__
  + ['mcmc']
  + ['plots', 'utils', 'stats', 'rednoise']
    )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
