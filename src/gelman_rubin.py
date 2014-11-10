# ******************************* START LICENSE *****************************
# 
# Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
# model-parameter best-fitting values and Bayesian posterior
# distributions.
# 
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington.  Principal developers included graduate student
# Patricio E. Cubillos and programmer Madison Stemm.  Statistical advice
# came from Thomas J. Loredo and Nate B. Lust.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# Our intent is to release this software under an open-source,
# reproducible-research license, once the code is mature and the first
# research paper describing the code has been accepted for publication
# in a peer-reviewed journal.  We are committed to development in the
# open, and have posted this code on github.com so that others can test
# it and give us feedback.  However, until its first publication and
# first stable release, we do not permit others to redistribute the code
# in either original or modified form, nor to publish work based in
# whole or in part on the output of this code.  By downloading, running,
# or modifying this code, you agree to these conditions.  We do
# encourage sharing any modifications with us and discussing them
# openly.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to:
# 
# Joseph Harrington <jh@physics.ucf.edu>
# Patricio Cubillos <pcubillos@fulbrightmail.org>
# 
# or alternatively,
# 
# Joseph Harrington and Patricio Cubillos
# UCF PSB 441
# 4111 Libra Drive
# Orlando, FL 32816-2385
# USA
# 
# Thank you for using MC3!
# ******************************* END LICENSE *******************************

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
