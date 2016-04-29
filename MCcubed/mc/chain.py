# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys
import os
import warnings
import random

import multiprocessing as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
import chisq as cs
import dwt   as dwt


# Ingnore RuntimeWarnings:
warnings.simplefilter("ignore", RuntimeWarning)

class Chain(mp.Process):
  """
  Background process.  This guy evaluates the model and calculates chisq.
  """
  def __init__(self, func, args, pipe, data, uncert,
               params, freepars, stepsize, pmin, pmax,
               walk, wlike, prior, priorlow, priorup, thinning,
               Z, Zsize, Zchisq, Zchain, M0, numaccept, outbounds,
               chainsize, bestp, bestchisq, ID, **kwds):
    """
    Class initializer.

    Parameters
    ----------
    func:  Callable
       Model fitting function.
    args:  List
       Additional arguments for function (besides the fitting parameters).
    pipe:  multiprocessing.Pipe object
       Pipe to communicate with mcmc.
    data:  1D shared-ctypes float ndarray
       Dependent data fitted by func.
    uncert:  1D Shared ctypes float ndarray
       Uncertainty of data.
    params:  1D float array
       Array of model parameters (including fixed and shared).
    freepars:  2D shared-ctypes float ndarray
       Current state of fitting parameters (X, as in Braak & Vrugt 2008).
    stepsize: 1D float ndarray
       Proposal jump scale.
    pmin: 1D float ndarray
       Lower boundaries of the posteriors.
    pmax: 1D float ndarray
       Upper boundaries of the posteriors.
    walk: String
       Flag to indicate wich MCMC algorithm to use [mrw, demc, snooker].
    wlike: Boolean
       Flag to use a wavelet-based likelihood function (True) or not (False).
    prior: 1D float ndarray
       Parameter prior.
    priorlow: 1D float ndarray
       Prior lower uncertainties.
    priorup: 1D float ndarray
       Prior uppper uncertainties.
    thinning: Integer
       Thinning factor of the chains.
    Z: 2D shared-ctype float ndarray
       MCMC parameters history (Z, as in Braak & Vrugt 2008).
    Zsize: Shared ctypes integer
       Current number of samples in the Z array.
    Zchisq: Float multiprocessing.Array
       Chi square values for the Z-array samples.
    Zchain: multiprocessing.Array integer
       Chain ID for the given state in the Z array.
    M0: Integer
       Initial number of samples in the Z array.
    numaccept: multiprocessing.Value integer
       Number of accepted MCMC proposals
    outbounds:  1D shared multiprocessing integer Array
       Array to count the number of out-of-bound proposals per free parameter.
    chainsize: multiprocessing.Array integer
       The current length of this chain.
    bestp: Shared ctypes float array
       The array with the current best-fitting parameter.
    bestchisq: Float multiprocessing.Value
       The chi-square value for bestp.
    ID: Integer
       Identification serial number for this chain.
    """
    # Multiprocessing setup:
    mp.Process.__init__(self, **kwds)
    self.daemon   = True     # FINDME: Understand daemon
    self.ID       = ID
    # MCMC setup:
    self.walk     = walk
    self.thinning = thinning
    self.Z        = Z
    self.Zsize    = Zsize
    self.Zchisq   = Zchisq
    self.Zchain   = Zchain
    self.chainsize = chainsize
    self.numaccept = numaccept
    self.outbounds = outbounds
    # Best values:
    self.bestp     = bestp
    self.bestchisq = bestchisq
    # Modeling function:
    self.func     = func
    self.args     = args
    # Model, fitting, and shared parameters:
    self.params   = params
    self.freepars = freepars
    self.stepsize = stepsize
    self.pmin     = pmin
    self.pmax     = pmax
    # Input/output Pipe:
    self.pipe     = pipe
    # Data:
    self.data     = data
    self.uncert   = uncert
    # Chisq function:
    self.wlike    = wlike

    # Index of parameters:
    self.ishare   = np.where(self.stepsize < 0)[0] # Shared parameter indices
    self.ifree    = np.where(self.stepsize > 0)[0] # Free parameter indices
    self.iprior   = np.where(priorlow != 0) # Indices of prior'ed parameters

    # Keep only the priors that count:
    self.prior    = prior   [self.iprior]
    self.priorlow = priorlow[self.iprior]
    self.priorup  = priorup [self.iprior]

    # Size of variables:
    self.nfree    = np.sum(self.stepsize > 0)      # Number of free parameters
    self.nchains  = np.shape(self.freepars)[0]
    self.Zlen     = np.shape(Z)[0]
    chainlen = (self.Zlen-M0) / self.nchains
    # Sample-index in Z-array to start this chain (for mrw and demc):
    self.index = M0 + (chainlen)*self.ID


  def run(self):
    """
    Process the requests queue until told to exit.
    """
    # Starting point:
    self.freepars[self.ID] = np.copy(self.Z[self.ID])
    chisq  = self.Zchisq[self.ID]
    nextp  = np.copy(self.params)  # Array for proposed sample
    nextchisq = 0.0                # Chi-square of nextp
    njump     = 0  # Number of jumps since last Z-update
    gamma  = 2.38 / np.sqrt(2*self.nfree)
    gamma2 = 0.0

    # The numpy random system must have its seed reinitialized in
    # each sub-processes to avoid identical 'random' steps.
    # random.randomint is process and thread safe.
    np.random.seed(random.randint(0,100000))

    # Run until completing the Z array:
    while True:
      njump += 1
      sjump = False  # Do a Snooker jump?
      normal = np.random.normal(0, self.stepsize[self.ifree], self.nfree)
      # Algorithm-specific proposals:
      if self.walk == "snooker":
        # Random sampling without replacement (0 <= iR1 != iR2 < Zsize):
        iR1 = np.random.randint(0,self.Zsize.value)
        iR2 = np.random.randint(1,self.Zsize.value)
        if iR2 == iR1:
          iR2 = 0
        sjump = np.random.uniform() < 0.1
        if sjump:
          # Snooker update:
          iz = np.random.randint(self.Zsize.value)
          z  = self.Z[iz]  # Not to confuse with Z!
          #z  = self.freepars[r1]
          dz = self.freepars[self.ID] - z
          zp1 = np.dot(self.Z[iR1], dz)
          zp2 = np.dot(self.Z[iR2], dz)
          jump = np.random.uniform(1.2, 2.2) * (zp1-zp2) * dz/np.dot(dz,dz)
        else: # Z update:
          jump = gamma*(self.Z[iR1] - self.Z[iR2]) + gamma2*normal
      elif self.walk == "mrw":
        jump = normal
      elif self.walk == "demc":
        b = self.pipe.recv()  # Synchronization
        # Select r1, r2 such that: r1 != r2 != ID:
        r1 = np.random.randint(1, self.nchains)
        if r1 == self.ID:
          r1 = 0
        # Pick r2 without replacement:
        r2 = (r1 + np.random.randint(2, self.nchains))%self.nchains
        if r2 == self.ID:
          r2 = (r1 + 1)%self.nchains
        jump = gamma*(self.freepars[r1] - self.freepars[r2]) + gamma2*normal

      # Propose next point:
      nextp[self.ifree] = np.copy(self.freepars[self.ID]) + jump

      # Check boundaries:
      outpars = np.asarray(((nextp < self.pmin) |
                            (nextp > self.pmax))[self.ifree])
      # If any of the parameter lied out of bounds, skip model evaluation:
      if np.any(outpars):
        self.outbounds[:] += outpars
      else:
        # Update shared parameters:
        for s in self.ishare:
          nextp[s] = nextp[-int(self.stepsize[s])-1]
        # Evaluate model:
        nextchisq = self.eval_model(nextp, ret="chisq")
        # Additional factor in Metropolis ratio for Snooker jump:
        if sjump:
          mrfactor = (np.linalg.norm(nextp[self.ifree]     -z) /
                      np.linalg.norm(self.freepars[self.ID]-z) )
        else:
          mrfactor = 1.0
        # Evaluate the Metropolis ratio:
        if np.exp(0.5 * (chisq - nextchisq)) * mrfactor > np.random.uniform():
          # Update freepars[ID]:
          self.freepars[self.ID] = np.copy(nextp[self.ifree])
          chisq = nextchisq
          with self.numaccept.get_lock():
            self.numaccept.value += 1
          # Check lowest chi-square:
          if chisq < self.bestchisq.value:
            self.bestp[self.ifree] = np.copy(self.freepars[self.ID])
            self.bestchisq.value = chisq

      # Update Z if necessary:
      if njump == self.thinning:
        # Update Z-array size:
        with self.Zsize.get_lock():
          # Stop when we fill Z:
          if self.Zsize.value == self.Zlen:
            break
          if self.walk == "snooker":
            self.index = self.Zsize.value
          self.Zsize.value += 1
        # Update values:
        self.Zchain[self.index] = self.ID
        self.Z     [self.index] = np.copy(self.freepars[self.ID])
        self.Zchisq[self.index] = chisq
        self.index += 1
        self.chainsize[self.ID] += 1
        njump = 0  # Reset njump
      if self.walk == "demc":
        self.pipe.send(chisq)


  def eval_model(self, params, ret="model"):
    """
    Evaluate the model for the requested set of parameters.

    Parameters
    ----------
    params: 1D float ndarray
       The set of model fitting parameters.
    ret: String
       Flag to indicate what to return.  Valid options:
       - 'model'  Return the evaluated model.
       - 'chisq'  Return chi-square.
       - 'both'   Return a list with the model and chisq.
    """
    if self.wlike:
      model = self.func(params[0:-3], *self.args)
    else:
      model = self.func(params, *self.args)

    # Calculate prioroff = params-prior:
    prioroff = params[self.iprior] - self.prior
    # Calculate chisq:
    if self.wlike:
      chisq = dwt.wlikelihood(params[-3:], model, self.data, prioroff,
                              self.priorlow, self.priorup)
    else:
      chisq = cs.chisq(model, self.data, self.uncert,
                       prioroff, self.priorlow, self.priorup)
    # Return evaluated model if requested:
    if   ret == "both":
      return [model, chisq]
    elif ret == "chisq":
      return chisq
    else:  # ret == "model"
      return model
