# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from .driver import *
from .gelman_rubin import *

from .driver import __all__ as dall
from .gelman_rubin import __all__ as gall

__all__ = dall + gall

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
