# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["gelman_rubin"]

import sys
import numpy as np

if sys.version_info.major == 2:
    range = xrange


def gelman_rubin(Z, Zchain, burnin):
    """
    Gelman--Rubin convergence test on a MCMC chain of parameters
    (Gelman & Rubin, 1992).

    Parameters
    ----------
    Z: 2D float ndarray
        A 2D array of shape (nsamples, npars) containing
        the parameter MCMC chains.
    Zchain: 1D integer ndarray
        A 1D array of length nsamples indicating the chain for each
        sample.
    burnin: Integer
        Number of iterations to remove.

    Returns
    -------
    GRfactor: 1D float ndarray
        The potential scale reduction factors of the chain for each
        parameter.  If they are much greater than 1, the chain is not
        converging.
    """
    # Number of chains:
    nchains = np.amax(Zchain) + 1
    # Number of free parameters:
    npars = np.shape(Z)[1]

    # Count number of samples in each chain:
    unique, nsamples = np.unique(Zchain, return_counts=True)
    # Remove pre-MCMC samples, and subtract burnin:
    nsamples = nsamples[unique >= 0] - burnin
    # Number of iterations (chain length):
    niter = np.amin(nsamples)
    if niter < 1:
        print("Not enough samples for Gelman-Rubin test.")
        return np.zeros(npars)

    # Reshape the Z array into a 3D array:
    data = np.zeros((nchains, niter, npars))
    for c in range(nchains):
        good = np.where(Zchain == c)[0][burnin:burnin+niter]
        data[c] = Z[good]

    # Allocate placeholder for results:
    GRfactor = np.zeros(npars)
    # Calculate psrf for each parameter:
    for i in range(npars):
        GRfactor[i] = psrf(data[:,:,i])
    return GRfactor


def psrf(chains):
    """
    Calculate the potential scale reduction factor (PSRF) of the
    Gelman and Rubin convergence test on a fitting parameter.

    Parameters
    ----------
    chains: 2D ndarray
        Array containing the chains for a single parameter.  Shape
        must be (nchains, chainlen).
    """
    # Get length of each chain and reshape:
    nchains, chainlen = np.shape(chains)

    # Calculate W (within-chain variance):
    W = np.mean(np.var(chains, axis=1))

    # Calculate B (between-chain variance):
    means = np.mean(chains, axis=1)
    mmean = np.mean(means)
    B     = (chainlen/(nchains-1.0)) * np.sum((means-mmean)**2)

    # Calculate V (posterior marginal variance):
    V = W*((chainlen - 1.0)/chainlen) + B*((nchains + 1.0)/(chainlen*nchains))

    # Calculate potential scale reduction factor (PSRF):
    rf = np.sqrt(V/W)

    return rf
