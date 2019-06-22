# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['mcmc', 'fit', 'mc', 'plots', 'utils', 'rednoise']

# Import sub-packages:
from . import fit
from . import mc
from . import plots
from . import utils
from . import rednoise

# Put the mcmc function at the package's top level:
mcmc = mc.mcmc

from . import VERSION as ver
__version__ = "{:d}.{:d}.{:d}".format(ver.MC3_VER, ver.MC3_MIN, ver.MC3_REV)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
