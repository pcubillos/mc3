# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["gelmanrubin"]

import numpy as np


def gelmanrubin(Z, Zchain, burnin):
    """
    Gelman & Rubin (1992) convergence test on a MCMC
    chain of parameters.

    Parameters
    ----------
    Z: 2D float ndarray
        A 2D array of shape (nsamples, nparameters) containing
        the parameter MCMC chains.
    Zchain: 1D integer ndarray
        A 1D array of length nsamples indicating the chain for each
        sample.
    burnin: Integer
        Number of iterations to remove.

    Returns
    -------
    GRfactor : 1D float ndarray
        The potential scale reduction factors of the chain for each
        parameter.  If they are much greater than 1, the chain is not
        converging.

    Uncredited developers
    ---------------------
    Chris Campo  (UCF)
    """
    # Number of chains:
    nchains = np.amax(Zchain) + 1
    # Number of free parameters:
    npars = np.shape(Z)[1]

    # Count number of samples in each chain:
    nsamples = np.zeros(nchains, np.int)
    for c in np.arange(nchains):
      nsamples[c] =  np.sum(Zchain == c)
    nsamples -= burnin
    # Number of iterations (chain length):
    niter = np.amin(nsamples)

    # Reshape the Z array into a 3D array:
    data = np.zeros((nchains, niter, npars))
    for c in np.arange(nchains):
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
