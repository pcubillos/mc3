__all__ = ["mcmc", "convergetest"]

from mcmc import mcmc
from gelman_rubin import convergetest

# For some reason the previous import leaves 'gelman_rubin' in the
# package's namespace.  Remove it manually:
del gelman_rubin
