import sys
import os
import time
import warnings

import multiprocessing as mp
import Queue as Queue
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/cfuncs/lib')
import chisq as cs
import dwt   as dwt

# Ingnore RuntimeWarnings:
warnings.simplefilter("ignore", RuntimeWarning)

class Chain(mp.Process):
  """
  Background process.  This guy evaluates the model, and calculates chisq.
  """
  def __init__(self, func, args, pipe, data, uncert, stepsize, pmin, pmax,
               walk, wlike, prior, priorlow, priorup, thinning,
               Z, Zsize, Zlen, Zchisq, Zchain, M0, numaccept,
               normal, unif, chainsize, bestp, bestchisq,
               ID, timeout, **kwds):
    """
    Class initializer.

    Parameters:
    -----------
    func: Callable
    args: List
    pipe: multiprocessing.Pipe
       Pipe to communicate with mcmc.
    data: Shared ctypes float ndarray
    uncert: Shared ctypes float ndarray
    stepsize: 1D float ndarray
    pmin: 1D float ndarray
    pmax: 1D float ndarray
    walk: String
       Flag to indicate the MCMC algorith to use.
    wlike: Boolean
       Flag to use a wavelet-based likelihood function (True) or not (False).
    prior: 1D float ndarray
    priorlow: 1D float ndarray
    priorup: 1D float ndarray
    thinning: Integer
    Z: 1D float shared ctype ndarray
       Flattened thinned MCMC parameters' sample
    Zsize: Shared ctypes integer
       Current number of samples in the Z array.
    Zlen: Integer
       Total number of samples in Z array.
    Zchisq: Float multiprocessing.Array
       Chi square values for the Z-array samples.
    Zchain: multiprocessing.Array integer
       Chain ID for the given state in the Z array.
    M0: Integer
       Initial number of samples in the Z array.
    numaccept: multiprocessing.Value integer
       Number of accepted MCMC proposals
    normal: 2D float ndarray
       A normal distribution [niter, nfree] for use by MRW or DEMC modes.
    unif: 1D float ndarray
       A uniform distribution to evaluate the Metropolis ratio.
    chainsize: multiprocessing.Array integer
       The current length of this chain.
    bestp: Shared ctypes float array
       The array with the current best-fitting parameter.
    bestchisq: Float multiprocessing.Value
       The chi-square value for bestp.
    ID: Integer
       Identification serial number for this chain.
    timeout: Float
       FINDME.
    """
    # Multiprocessing setup:
    mp.Process.__init__(self, **kwds)
    self.daemon   = True     # FINDME: Understand daemon
    self.ID       = ID
    self.timeout  = timeout  # FINDME: Keep?
    # MCMC setup:
    self.walk     = walk
    self.thinning = thinning
    self.Z        = Z
    self.Zsize    = Zsize
    self.Zlen     = Zlen
    self.Zchisq   = Zchisq
    self.Zchain   = Zchain
    self.chainsize = chainsize
    self.normal   = normal
    self.unif     = unif
    self.outbounds = 0  # FINDME: get as shared ctypes array input.
    self.numaccept = numaccept
    self.chainlen = len(unif) # Number of iterations for this chain.
    # Best values:
    self.bestp     = bestp
    self.bestchisq = bestchisq
    # Modeling function:
    self.func     = func
    self.args     = args
    # Model, fitting, and shared parameters:
    self.stepsize = stepsize
    self.ishare   = np.where(self.stepsize < 0)[0] # Shared parameter indices
    self.ifree    = np.where(self.stepsize > 0)[0] # Free parameter indices
    self.nfree    = np.sum(self.stepsize > 0)      # Number of free parameters
    self.pmin     = pmin
    self.pmax     = pmax
    # Input/output Pipe:
    self.pipe     = pipe
    # Data:
    self.data     = data
    self.uncert   = uncert
    # Chisq function:
    self.wlike    = wlike
    # Priors:
    self.iprior   = np.where(priorlow != 0) # Indices of prior'ed parameters
    self.prior    = prior   [self.iprior]  # Keep only the ones that count
    self.priorlow = priorlow[self.iprior]
    self.priorup  = priorup [self.iprior]

    # Sample-index in Z-array to start this chain:
    self.index = M0 + (self.chainlen/self.thinning)*self.ID
    #if self.ID == 0:
    #  print("Chainsize {:2d}, chainlen {:d}".format(self.chainsize[self.ID],
    #                                                self.chainlen))
    #print("Chain {:2d} has index {:d}".format(self.ID, self.index))

    # FINDME: Do I need some custom initialization?
    if   self.walk == "mrw":
      pass
    elif self.walk == "demc":
      pass
    elif self.walk == "snooker":
      pass


  def run(self):
    """
    Process the requests queue until told to exit.
    """
    # Starting point:
    # FINDME: Reshape Z to 2D, Then I could simply use self.index.
    params = self.Z[self.ID*self.nfree:(self.ID+1)*self.nfree]
    chisq  = self.Zchisq[self.ID]
    nextp  = np.copy(params)  # Allocate array for proposed sample
    nextchisq = 0.0           # Chi-square of nextp
    njump     = 0  # Number of jumps since last Z-update
    niter     = 0  # Current number of iterations

    # Run until completing the Z array:
    while self.Zsize.value < self.Zlen:
      #print("FLAG 010: niter={:3d}".format(niter))
      njump += 1
      # Algorithm-specific proposals:
      if self.walk == "snooker":
        # Snooker update:
        pass
        # HACKY trick:
        if niter == self.chainlen:
          niter = 0
      else:
        # Stop when we complete chainlen iterations:
        if niter == self.chainlen:
          #print("Chainsize {} is {}".format(self.ID, self.chainsize[self.ID]))
          #print("Index     {} is {}".format(self.ID, self.index))
          #print("Zsize {}".format(self.Zsize.value))
          break
        if self.walk == "mrw":
          jump = self.normal[niter]
        elif self.walk == "demc":
          jump = self.pipe.recv()

      # Propose next point:
      nextp[self.ifree] = np.copy(params[self.ifree]) + jump

      # Check boundaries:
      outpars = np.asarray(((nextp < self.pmin) |
                            (nextp > self.pmax))[self.ifree])
      # FINDME: be careful
      # If any of the parameter lied out of bounds, skip model evaluation:
      if np.any(outpars):
        self.outbounds[self.ID] += 1
        #if self.walk == "demc":
        #  self.pipe.send(-1)
        #continue
      else:
        # Update shared parameters:
        for s in self.ishare:
          nextp[s] = nextp[-int(self.stepsize[s])-1]
        # Evaluate model:
        nextchisq = self.eval_model(nextp)
        # Evaluate the Metropolis ratio:
        if np.exp(0.5 * (chisq - nextchisq)) > self.unif[niter]:
          params = np.copy(nextp)
          chisq = nextchisq
          with self.numaccept.get_lock():
            self.numaccept.value += 1
          # Check lowest chi-square:
          if chisq < self.bestchisq.value:
            self.bestp[:] = np.copy(params)
            self.bestchisq.value = chisq

      # Update to Z if necessary:
      if njump == self.thinning:
        # Update Z-array size:
        with self.Zsize.get_lock():
          if self.walk == "snooker":
            self.index = self.Zsize.value
          # FINDME: Should I put an else here?
          self.Zsize.value += 1
        # Update values:
        self.Zchain[self.index] = self.ID
        self.Z[self.index*self.nfree:(self.index+1)*self.nfree] = params
        self.Zchisq[self.index] = chisq
        #else:
        self.index += 1
        self.chainsize[self.ID] += 1
        njump = 0  # Reset njump
      if self.walk == "demc":
        self.pipe.send(chisq)
      niter += 1


  def eval_model(self, params, retmodel=False):
    """
    Evaluate the model for the requested set of parameters.

    Parameters:
    -----------
    params: 1D float ndarray
       The set of model fitting parameters.
    retmodel: Boolean
       If True, also return the evaluated model.  Else, just return chi-square.
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
    if retmodel:
      return [model, chisq]
    return  chisq
