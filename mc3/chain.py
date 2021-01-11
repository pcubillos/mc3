# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys
import warnings
import random
import multiprocessing as mp

import numpy as np

from . import stats as ms

if sys.version_info.major == 2:
    range = xrange

# Ingnore RuntimeWarnings:
warnings.simplefilter("ignore", RuntimeWarning)


class Chain(mp.Process):
  """
  Background process.  This guy evaluates the model and calculates chisq.
  """
  def __init__(self, func, args, pipe, data, uncert,
               params, freepars, pstep, pmin, pmax,
               sampler, wlike, prior, priorlow, priorup, thinning,
               fgamma, fepsilon, Z, zsize, log_post, zchain, M0,
               numaccept, outbounds, ncpp,
               chainsize, bestp, best_log_post, ID, ncpu, **kwds):
      """
      Chain class initializer.

      Parameters
      ----------
      func: Callable
          Model fitting function.
      args: List
          Additional arguments for function (besides the fitting parameters).
      pipe: multiprocessing.Pipe object
          Pipe to communicate with mcmc.
      data: 1D shared-ctypes float ndarray
          Dependent data fitted by func.
      uncert: 1D Shared ctypes float ndarray
          Uncertainty of data.
      params: 1D float array
          Array of model parameters (including fixed and shared).
      freepars: 2D shared-ctypes float ndarray
          Current state of fitting parameters (X, as in Braak & Vrugt 2008).
      pstep: 1D float ndarray
          Proposal jump scale.
      pmin: 1D float ndarray
          Lower boundaries of the posteriors.
      pmax: 1D float ndarray
          Upper boundaries of the posteriors.
      sampler: String
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
      fgamma: Float
          Proposals jump scale factor for DEMC's gamma.
          The code computes: gamma = fgamma * 2.38 / sqrt(2*Nfree)
      fepsilon: Float
          Jump scale factor for DEMC's support distribution.
          The code computes: e = fepsilon * Normal(0, pstep)
      Z: 2D shared-ctype float ndarray
          MCMC parameters history (Z, as in Braak & Vrugt 2008).
      zsize: Shared ctypes integer
          Current number of samples in the Z array.
      log_post: Float multiprocessing.Array
          log(posterior) values for the samples in Z.
      zchain: multiprocessing.Array integer
          Chain ID for the given state in the Z array.
      M0: Integer
          Initial number of samples in the Z array.
      numaccept: multiprocessing.Value integer
          Number of accepted MCMC proposals
      outbounds:  1D shared multiprocessing integer Array
          Number of out-of-bound proposals per free parameter.
      ncpp: Integer
          Number of chains for this process.
      chainsize: multiprocessing.Array integer
          The current length of this chain.
      bestp: Shared ctypes float array
          The array with the current best-fitting parameter.
      best_log_post: Float multiprocessing.Value
          The log(posterior) value for bestp.
      ID: Integer
          Identification serial number for this chain.
      ncpu: Integer
          The number of processes running chains.
      """
      # Multiprocessing setup:
      mp.Process.__init__(self, **kwds)
      self.daemon   = True
      self.ID       = ID
      self.ncpp     = ncpp
      self.ncpu     = ncpu
      # MCMC setup:
      self.sampler  = sampler
      self.thinning = thinning
      self.fgamma   = fgamma
      self.fepsilon = fepsilon
      self.Z        = Z
      self.zsize    = zsize
      self.log_post = log_post
      self.zchain   = zchain
      self.chainsize = chainsize
      self.M0        = M0
      self.numaccept = numaccept
      self.outbounds = outbounds
      # Best values:
      self.bestp     = bestp
      self.best_log_post = best_log_post
      # Modeling function:
      self.func     = func
      self.args     = args
      # Model, fitting, and shared parameters:
      self.params   = params
      self.freepars = freepars
      self.pstep    = pstep
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
      self.ishare   = np.where(self.pstep < 0)[0]
      self.ifree    = np.where(self.pstep > 0)[0]

      # Keep only the priors that count:
      self.prior    = prior
      self.priorlow = priorlow
      self.priorup  = priorup

      # Size of variables:
      self.nfree    = np.sum(self.pstep > 0)   # Number of free parameters
      self.nchains  = np.shape(self.freepars)[0]
      self.Zlen     = np.shape(Z)[0]

      # Length of mrw/demc chains:
      self.chainlen = int((self.Zlen) / self.nchains)


  def run(self):
      """
      Process the requests queue until terminated.
      """
      # Indices in Z-array to start this chains:
      IDs = np.arange(self.ID, self.nchains, self.ncpu)
      self.index = self.M0 + IDs
      for j in range(self.ncpp):
          if np.any(self.zchain==self.ID):  # (i.e., resume=True)
              # Set ID to the last iteration for this chain:
              IDs[j] = self.index[j] = np.where(self.zchain==IDs[j])[0][-1]
          self.freepars[self.ID + j*self.ncpu] = np.copy(self.Z[IDs[j]])
      chisq = -2*self.log_post[IDs]

      nextp     = np.copy(self.params)  # Array for proposed sample
      nextchisq = 0.0                   # Chi-square of nextp
      njump     = 0                     # Number of jumps since last Z-update
      gamma     = self.fgamma * 2.38 / np.sqrt(2*self.nfree)

      # The numpy random system must have its seed reinitialized in
      # each sub-processes to avoid identical 'random' steps.
      # random.randomint is process- and thread-safe.
      np.random.seed(random.randint(0, 100000))

      # Run until completing the Z array:
      while True:
          njump += 1
          normal = np.random.normal(0, self.pstep[self.ifree], self.nfree)

          if self.sampler == "demc":
              b = self.pipe.recv()  # Synchronization flag

          for j in range(self.ncpp):
              ID = self.ID + j*self.ncpu
              mrfactor = 1.0

              # Algorithm-specific proposals jumps:
              if self.sampler == "snooker":
                  # Sampling without replacement (0 <= iR1 != iR2 < zsize):
                  iR1 = np.random.randint(0, self.zsize.value)
                  iR2 = np.random.randint(1, self.zsize.value)
                  if iR2 == iR1:
                      iR2 = 0
                  sjump = np.random.uniform() < 0.1
                  if sjump:  # Snooker update:
                      iz = np.random.randint(self.zsize.value)
                      z  = self.Z[iz]  # Not to confuse with Z!
                      if np.all(z == self.freepars[ID]):  # Do not project:
                          jump = np.random.uniform(1.2, 2.2) \
                                     * (self.Z[iR2]-self.Z[iR1])
                      else:
                          dz = self.freepars[ID] - z
                          zp1 = np.dot(self.Z[iR1], dz)
                          zp2 = np.dot(self.Z[iR2], dz)
                          jump = np.random.uniform(1.2, 2.2) * (zp1-zp2) \
                                     * dz/np.dot(dz,dz)
                  else: # Z update:
                      jump = gamma*(self.Z[iR1] - self.Z[iR2]) \
                                 + self.fepsilon*normal

              elif self.sampler == "mrw":
                  jump = normal
              elif self.sampler == "demc":
                  # Select r1, r2 such that: r1 != r2 != ID:
                  r1 = np.random.randint(1, self.nchains)
                  if r1 == ID:
                      r1 = 0
                  # Pick r2 without replacement:
                  r2 = (r1 + np.random.randint(2, self.nchains))%self.nchains
                  if r2 == ID:
                      r2 = (r1 + 1) % self.nchains
                  jump = gamma*(self.freepars[r1] - self.freepars[r2]) \
                         + self.fepsilon*normal

              # Propose next point:
              nextp[self.ifree] = np.copy(self.freepars[ID]) + jump

              # Check boundaries:
              outpars = np.asarray(((nextp < self.pmin) |
                                    (nextp > self.pmax))[self.ifree])
              # If any parameter lied out of bounds, skip model evaluation:
              if np.any(outpars):
                  self.outbounds[:] += outpars
              else:
                  # Update shared parameters:
                  for s in self.ishare:
                      nextp[s] = nextp[-int(self.pstep[s])-1]
                  # Evaluate model:
                  nextchisq = self.eval_model(nextp, ret="chisq")
                  # Additional factor in Metropolis ratio for Snooker jump:
                  if self.sampler == "snooker" and sjump:
                      # squared norm of current and next:
                      cnorm = np.dot(self.freepars[ID]-z, self.freepars[ID]-z)
                      nnorm = np.dot(nextp[self.ifree]-z, nextp[self.ifree]-z)
                      mrfactor = (nnorm/cnorm)**(0.5*(self.nfree-1))
                  # Evaluate the Metropolis ratio:
                  if np.exp(0.5*(chisq[j]-nextchisq)) * mrfactor \
                     > np.random.uniform():
                      # Update freepars[ID]:
                      self.freepars[ID] = np.copy(nextp[self.ifree])
                      chisq[j] = nextchisq
                      with self.numaccept.get_lock():
                          self.numaccept.value += 1
                      # Check lowest chi-square:
                      if chisq[j] < -2*self.best_log_post.value:
                          self.bestp[self.ifree] = np.copy(self.freepars[ID])
                          self.best_log_post.value = -0.5*chisq[j]
              # Update Z if necessary:
              if njump == self.thinning:
                  with self.zsize.get_lock():
                      # Stop when we fill Z:
                      if self.zsize.value == self.Zlen:
                          return
                      if self.sampler == "snooker":
                          self.index[j] = self.zsize.value
                      self.zsize.value += 1
                  # Update values:
                  self.zchain[self.index[j]] = ID
                  self.Z     [self.index[j]] = np.copy(self.freepars[ID])
                  self.log_post[self.index[j]] = -0.5*chisq[j]
                  self.index[j] += self.nchains
                  self.chainsize[ID] += 1

          if njump == self.thinning:
              njump = 0  # Reset njump

          if self.sampler == "demc":
              self.pipe.send(chisq[j])
          # Stop when the chain is complete:
          if self.sampler in ["mrw","demc"] \
              and self.chainsize[0]==self.chainlen:
              return


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

      # Reject proposed iteration if any model value is infinite:
      if np.any(model == np.inf):
          chisq = np.inf
      else:
          # Calculate chisq:
          if self.wlike:
              chisq = ms.dwt_chisq(model, self.data, params,
                  self.prior, self.priorlow, self.priorup)
          else:
              chisq = ms.chisq(model, self.data, self.uncert,
                  params, self.prior, self.priorlow, self.priorup)

      if ret == "both":
          return [model, chisq]
      elif ret == "chisq":
          return chisq
      else:  # ret == "model"
          return model
