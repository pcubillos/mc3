import numpy as np

def convergetest(chains):
    """
    Wrapper for the Gelman & Rubin (1992) convergence test on a MCMC
    chain parameters.

    Parameters
    ----------
    chains : ndarray
        A 3D array of shape (nchains, nparameters, chainlen) containing
        the parameter MCMC chains.

    Returns
    -------
    psrf : ndarray
        The potential scale reduction factors of the chain.  If the
        chain has converged, each value should be close to unity.  If
        they are much greater than 1, the chain has not converged and
        requires more samples.  The order of psrfs in this vector are
        in the order of the free parameters.

    Modification History:
    ---------------------
    2010-08-20  ccampo    Initial version.
    2014-03-31  patricio  Re-adapted to work with mcmc.
    """    
    # Allocate placeholder for results:
    npars = np.shape(chains)[1]
    psrf = np.zeros(npars)

    # Calculate psrf for each parameter:
    for i in range(npars):
      psrf[i] = gelmanrubin(chains[:, i, :])
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
