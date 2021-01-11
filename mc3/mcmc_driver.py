# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["mcmc"]

import sys
import time
import ctypes
import multiprocessing as mpr

if sys.version_info.major == 2:
    range = xrange

import numpy as np

from . import chain as ch
from . import utils as mu
from . import stats as ms


def mcmc(data, uncert, func, params, indparams, pmin, pmax, pstep,
         prior, priorlow, priorup, nchains, ncpu, nsamples, sampler,
         wlike, fit_output, grtest, grbreak, grnmin, burnin, thinning,
         fgamma, fepsilon, hsize, kickoff, savefile, resume, log):
  """
  Mid-level routine called by mc3.sample() to execute Markov-chain Monte
  Carlo run.

  Parameters
  ----------
  data: 1D float ndarray
      Data to be fit by func.
  uncert: 1D float ndarray
      Uncertainties of data.
  func: Callable or string-iterable
      The callable function that models data as:
      model = func(params, *indparams)
  params: 1D float ndarray
      Set of initial fitting parameters for func.
  indparams: tuple
      Additional arguments required by func.
  pmin: 1D ndarray
      Lower boundaries for the posterior exploration.
  pmax: 1D ndarray
      Upper boundaries for the posterior exploration.
  pstep: 1D ndarray
      Parameter stepping.
  prior: 1D ndarray
      Parameter prior distribution means.
  priorlow: 1D ndarray
      Lower prior uncertainty values.
  priorup: 1D ndarray
      Upper prior uncertainty values.
  nchains: Scalar
      Number of simultaneous chains to run.
  ncpu: Integer
      Number of processors for the MCMC chains.
  nsamples: Scalar
      Total number of samples.
  sampler: String
      MCMC sampling algorithm select from [mrw, demc, snooker]
  wlike: Bool
      If True, calculate the likelihood in a wavelet-base.
  grtest: Boolean
      Run Gelman & Rubin test.
  grbreak: Float
      Gelman-Rubin convergence threshold to stop the MCMC.
  grnmin: Float
      Minimum number of samples required for grbreak to stop the MCMC.
  burnin: Integer
      Number of burned-in (discarded) iterations.
  thinning: Integer
      Thinning factor of the chains (use every thinning-th iteration).
  fgamma: Float
      Proposals jump scale factor for DEMC's gamma.
  fepsilon: Float
      Jump scale factor for DEMC's support distribution.
  hsize: Integer
      Number of initial samples per chain.
  kickoff: String
      How to start the chains:
      'normal' for normal distribution around initial guess, or
      'uniform' for uniform distribution withing the given boundaries.
  savefile: String
      If not None, filename to store allparams and other MCMC results.
  resume: Boolean
      If True resume a previous run.
  log: mc3.utils.Log instance
      Logging object.

  Returns
  -------
  output: Dict
      A Dictionary containing the MCMC posterior distribution and related
      stats, including:
      - posterior: thinned posterior distribution of shape [nsamples, nfree].
      - zchain: chain indices for each sample in Z.
      - zmask: indices that turn Z into the desired posterior (remove burn-in).
      - chisq: chi^2 value for each sample in Z.
      - log_posterior: log(posterior) for the samples in Z.
      - burnin: number of burned-in samples per chain.
      - bestp: model parameters for the optimal log(posterior) sample.
      - best_model: model evaluated at bestp.
      - best_chisq: chi^2 for the optimal log(posterior) in the sample.
      - best_log_post: optimal log(posterior) in posterior.
      - acceptance_rate: sample's acceptance rate.

  Examples
  --------
  >>> # See https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html
  """
  nfree  = int(np.sum(pstep > 0))
  ifree  = np.where(pstep > 0)[0]
  ishare = np.where(pstep < 0)[0]

  if resume:
      oldrun = np.load(savefile)
      zold = oldrun["Z"]
      zchain_old = oldrun["zchain"]
      # Size of posterior (prior to this MCMC sample):
      pre_zsize = np.shape(zold)[0]
  else:
      pre_zsize = M0 = hsize*nchains

  # Number of Z samples per chain:
  nzchain = int(np.ceil(nsamples/nchains/thinning))
  # Number of iterations per chain:
  niter = nzchain * thinning
  # Total number of Z samples (initial + chains):
  zlen = pre_zsize + nzchain*nchains

  if not resume and niter < burnin:
      log.error("The number of burned-in samples ({:d}) is greater than "
                "the number of iterations per chain ({:d}).".
                format(burnin, niter))

  # Initialize shared-memory variables:
  sm_freepars = mpr.Array(ctypes.c_double, nchains*nfree)
  freepars = np.ctypeslib.as_array(sm_freepars.get_obj())
  freepars = freepars.reshape((nchains, nfree))

  best_log_post = mpr.Value(ctypes.c_double, np.inf)
  sm_bestp = mpr.Array(ctypes.c_double, np.copy(params))
  bestp = np.ctypeslib.as_array(sm_bestp.get_obj())
  # There seems to be a strange behavior with np.ctypeslib.as_array()
  # when the argument is a single-element array. In this case, the
  # returned value is a two-dimensional array, instead of 1D. The
  # following line fixes(?) that behavior:
  if np.ndim(bestp) > 1:
      bestp = bestp.flatten()

  numaccept = mpr.Value(ctypes.c_int, 0)
  outbounds = mpr.Array(ctypes.c_int, nfree)

  # Z array with the chains history:
  sm_Z = mpr.Array(ctypes.c_double, zlen*nfree)
  Z = np.ctypeslib.as_array(sm_Z.get_obj())
  Z = Z.reshape((zlen, nfree))

  # Chi-square value of Z:
  sm_log_post = mpr.Array(ctypes.c_double, zlen)
  log_post = np.ctypeslib.as_array(sm_log_post.get_obj())
  # Chain index for given state in the Z array:
  sm_zchain = mpr.Array(ctypes.c_int, -np.ones(zlen, np.int))
  zchain = np.ctypeslib.as_array(sm_zchain.get_obj())
  # Current number of samples in the Z array:
  zsize = mpr.Value(ctypes.c_int, M0)
  # Burned samples in the Z array per chain:
  zburn = int(burnin/thinning)

  # Include values from previous run:
  if resume:
      Z[0:pre_zsize,:] = zold
      zchain[0:pre_zsize] = oldrun["zchain"]
      log_post[0:pre_zsize] = oldrun["log_post"]
      # Redefine zsize:
      zsize.value = pre_zsize
      numaccept.value = int(oldrun["numaccept"])

  # Set GR N-min as fraction if needed:
  if grnmin > 0 and grnmin < 1:
      grnmin = int(grnmin*(zlen-M0-zburn*nchains))
  elif grnmin < 0:
      log.error("Invalid 'grnmin' argument (minimum number of samples to "
          "stop the MCMC under GR convergence), must either be grnmin > 1"
          "to set the minimum number of samples, or 0 < grnmin < 1"
          "to set the fraction of samples required to evaluate.")
  # Add these to compare grnmin to zsize (which also include them):
  grnmin += int(M0 + zburn*nchains)

  # Current length of each chain:
  sm_chainsize = mpr.Array(ctypes.c_int, np.tile(hsize, nchains))
  chainsize = np.ctypeslib.as_array(sm_chainsize.get_obj())

  # Number of chains per processor:
  ncpp = np.tile(int(nchains/ncpu), ncpu)
  ncpp[0:nchains % ncpu] += 1

  # Launch Chains:
  pipes  = []
  chains = []
  for i in range(ncpu):
      p = mpr.Pipe()
      pipes.append(p[0])
      chains.append(ch.Chain(func, indparams, p[1], data, uncert,
          params, freepars, pstep, pmin, pmax,
          sampler, wlike, prior, priorlow, priorup, thinning,
          fgamma, fepsilon, Z, zsize, log_post, zchain, M0,
          numaccept, outbounds, ncpp[i],
          chainsize, bestp, best_log_post, i, ncpu))

  if resume:
      bestp = oldrun['bestp']
      best_log_post.value = oldrun['best_log_post']
      for c in range(nchains):
          chainsize[c] = np.sum(zchain_old==c)
  else:
      # Populate the M0 initial samples of Z:
      Z[0] = np.clip(params[ifree], pmin[ifree], pmax[ifree])
      for j, idx in enumerate(ifree):
          if kickoff == "normal":   # Start with a normal distribution
              vals = np.random.normal(params[idx], pstep[idx], M0-1)
              # Stay within pmin and pmax boundaries:
              vals[np.where(vals < pmin[idx])] = pmin[idx]
              vals[np.where(vals > pmax[idx])] = pmax[idx]
              Z[1:M0,j] = vals
          elif kickoff == "uniform":  # Start with a uniform distribution
              Z[1:M0,j] = np.random.uniform(pmin[idx], pmax[idx], M0-1)

      # Evaluate models for initial sample of Z:
      fitpars = np.asarray(params)
      for i in range(M0):
          fitpars[ifree] = Z[i]
          # Update shared parameters:
          for s in ishare:
              fitpars[s] = fitpars[-int(pstep[s])-1]
          log_post[i] = -0.5*chains[0].eval_model(fitpars, ret="chisq")

      # Best-fitting values (so far):
      izbest = np.argmax(log_post[0:M0])
      best_log_post.value = log_post[izbest]
      bestp[ifree] = np.copy(Z[izbest])
      if fit_output is not None:
          bestp[:] = np.copy(fit_output['bestp'])
          best_log_post.value = fit_output['best_log_post']

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Start loop:
  print("Yippee Ki Yay Monte Carlo!")
  log.msg("Start MCMC chains  ({:s})".format(time.ctime()))
  for chain in chains:
      chain.start()
  bit = bool(1)  # Dummy variable to send through pipe for DEMC
  # Intermediate steps to run GR test and print progress report:
  intsteps = (nzchain*nchains) / 10
  report = intsteps

  while True:
      # Proposal jump:
      if sampler == "demc":
          # Send and receive bit for DEMC synchronization:
          for pipe in pipes:
              pipe.send(bit)
          for pipe in pipes:
              b = pipe.recv()

      # Print intermediate info:
      if (zsize.value-pre_zsize >= report) or (zsize.value == zlen):
          report += intsteps
          log.progressbar((zsize.value+1.0-pre_zsize)/(nzchain*nchains))

          log.msg("Out-of-bound Trials:\n{:s}".
                  format(str(np.asarray(outbounds[:]))),      width=80)
          log.msg("Best Parameters: (chisq={:.4f})\n{:s}".
                  format(-2*best_log_post.value, str(bestp[ifree])), width=80)

          ## Save current results:
          #if savefile is not None:
          #    np.savez(savefile, Z=Z, zchain=zchain)

          # Gelman-Rubin statistics:
          if grtest and np.all(chainsize > (zburn+hsize)):
              psrf = ms.gelman_rubin(Z, zchain, zburn)
              log.msg("Gelman-Rubin statistics for free parameters:\n{:s}".
                       format(str(psrf)), width=80)
              if np.all(psrf < 1.01):
                  log.msg("All parameters converged to within 1% of unity.")
              if (grbreak > 0.0 and np.all(psrf < grbreak) and
                  zsize.value > grnmin):
                  with zsize.get_lock():
                      zsize.value = zlen
                  log.msg("\nAll parameters satisfy the GR convergence "
                      "threshold of {:g}, stopping the MCMC.".format(grbreak))
                  break
          if zsize.value == zlen:
              break

  for chain in chains:  # Make sure to terminate the subprocesses
      chain.terminate()

  # Evaluate model for best fitting parameters:
  fitpars = np.asarray(params)
  fitpars[ifree] = np.copy(bestp[ifree])
  for s in ishare:
      fitpars[s] = fitpars[-int(pstep[s])-1]
  best_model = chains[0].eval_model(fitpars)

  # Remove pre-MCMC and post-MCMC alocated samples:
  zvalid = zchain>=0
  Z = Z[zvalid]
  zchain = zchain[zvalid]
  log_post = log_post[zvalid]
  log_prior = ms.log_prior(Z, prior, priorlow, priorup, pstep)
  chisq = -2*(log_post - log_prior)
  best_log_prior = ms.log_prior(bestp[ifree], prior, priorlow, priorup, pstep)
  best_chisq = -2*(best_log_post.value - best_log_prior)
  # And remove burn-in samples:
  posterior, _, zmask = mu.burn(Z=Z, zchain=zchain, burnin=zburn)

  # Number of evaluated and kept samples:
  nsample  = len(Z)*thinning
  nzsample = len(posterior)

  # Print out Summary:
  log.msg('\nMCMC Summary:'
          '\n-------------')
  fmt = len(str(nsample))
  log.msg("Number of evaluated samples:        {:{}d}".
          format(nsample,  fmt), indent=2)
  log.msg("Number of parallel chains:          {:{}d}".
          format(nchains,  fmt), indent=2)
  log.msg("Average iterations per chain:       {:{}d}".
          format(nsample//nchains, fmt), indent=2)
  log.msg("Burned-in iterations per chain:     {:{}d}".
          format(burnin,   fmt), indent=2)
  log.msg("Thinning factor:                    {:{}d}".
          format(thinning, fmt), indent=2)
  log.msg("MCMC sample size (thinned, burned): {:{}d}".
          format(nzsample, fmt), indent=2)
  log.msg("Acceptance rate:   {:.2f}%\n".
          format(numaccept.value*100.0/nsample), indent=2)

  # Build the output dict:
  output = {
      # The posterior:
      'posterior':Z,
      'zchain':zchain,
      'chisq':chisq,
      'log_post':log_post,
      'zmask':zmask,
      'burnin':zburn,
      # Posterior stats:
      'acceptance_rate':numaccept.value*100.0/nsample,
      # Best-fit stats:
      'bestp':bestp,
      'best_model':best_model,
      'best_log_post':best_log_post.value,
      'best_chisq':best_chisq,
  }
  return output
