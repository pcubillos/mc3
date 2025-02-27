# Copyright (c) 2015-2025 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

from .plot_functions import *
from .posterior import *
from .colors import *

__all__ = (
    plot_functions.__all__
    + posterior.__all__
    + colors.__all__
)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
