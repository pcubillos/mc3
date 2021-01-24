# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from .gelman import *
from .stats import *
from .time_averaging import *
from .prayer import *

__all__ = (
    gelman.__all__
  + stats.__all__
  + time_averaging.__all__
  + prayer.__all__
    )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
