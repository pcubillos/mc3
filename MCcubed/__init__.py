# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['mcmc', 'fit', 'mc', 'plots', 'utils', 'rednoise']

# Import sub-packages:
from . import fit
from . import mc
from . import plots
from . import utils
from . import rednoise

# Import MCMC function:
from .mc.driver import mcmc

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
