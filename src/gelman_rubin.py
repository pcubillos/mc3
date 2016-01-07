# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import numpy as np

def convergetest(Z, Zchain, burnin):
    """
    Wrapper for the Gelman & Rubin (1992) convergence test on a MCMC
    chain of parameters.

    Parameters
    ----------
    Z : ndarray
        A 2D array of shape (nsamples, nparameters) containing
        the parameter MCMC chains.
    Zchain: Integer ndarray
        A 1D array of length nsamples indicating the chain for each
        sample.
    burnin: Integer
        Number of iterations to remove.

    Returns
    -------
    psrf : ndarray
        The potential scale reduction factors of the chain.  If the
        chain has converged, each value should be close to unity.  If
        they are much greater than 1, the chain has not converged and
        requires more samples.
    Developer team
    --------------
    Chris Campo        University of Central Florida.
    Patricio Cubillos  Space Research Institute, Graz, Austria.
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
      good = np.where(Zchain == c)[0][burnin:]
      data[c] = Z[good]

    # Allocate placeholder for results:
    psrf = np.zeros(npars)
    # Calculate psrf for each parameter:
    for i in range(npars):
      psrf[i] = gelmanrubin(data[:, :, i])
    return psrf


def gelmanrubin(chains):
    """
    Calculate the potential scale reduction factor of the Gelman & Rubin
    convergence test on a fitting parameter

    Parameters:
    -----------
    chains: 2D ndarray
       Array containing the chains for a single parameter.  Shape
       must be (nchains, chainlen)

    Modification History:
    ---------------------
    2014-03-31  patricio  Added documentation.
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
    psrf = np.sqrt(V/W)

    return psrf
