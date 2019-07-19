# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from .mcmc_driver import *
from .fit_model import *
from . import plots
from . import utils
from . import rednoise
from . import VERSION as ver


__version__ = "{:d}.{:d}.{:d}".format(ver.MC3_VER, ver.MC3_MIN, ver.MC3_REV)

__all__ = (
    mcmc_driver.__all__
  + fit_model.__all__
  + ['plots', 'utils', 'rednoise']
    )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
