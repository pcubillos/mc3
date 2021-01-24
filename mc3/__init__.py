# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from .sampler_driver import *
from .fit_driver import *
from . import plots
from . import utils
from . import stats
from .VERSION import __version__


__all__ = (
    sampler_driver.__all__
  + fit_driver.__all__
  + ['plots', 'utils', 'stats']
    )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
