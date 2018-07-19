# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ =["mcmc"]

import os
import sys
import time
import importlib
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mpr

from .  import gelman_rubin as gr
from .  import chain   as ch

from .. import fit     as mf
from .. import utils   as mu
from .. import plots   as mp
from .. import VERSION as ver

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
import timeavg  as ta


def mcmc(data,          uncert=None,    func=None,      indparams=[],
         params=None,   pmin=None,      pmax=None,      stepsize=None,
         prior=None,    priorlow=None,  priorup=None,
         nchains=10,    nproc=None,     nsamples=10,    walk='demc',
         wlike=False,   leastsq=True,   lm=False,       chisqscale=False,
         grtest=True,   grbreak=0.01,   grnmin=0.5,
         burnin=0,      thinning=1,
         fgamma=1.0,    fepsilon=0.0,   hsize=1,        kickoff='normal',
         plots=False,   ioff=False,     showbp=True,
         savefile=None, savemodel=None, resume=False,
         rms=False,     log=None,       parname=None,   full_output=False,
         chireturn=False):
  """
  This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

  Parameters
  ----------
  data: 1D ndarray
     Dependent data fitted by func.
  uncert: 1D ndarray
     Uncertainty of data.
  func: callable or string-iterable
     The callable function that models data as:
        model = func(params, *indparams)
     Or an iterable (list, tuple, or ndarray) of 3 strings:
        (funcname, modulename, path)
     that specify the function name, function module, and module path.
     If the module is already in the python-path scope, path can be omitted.
  indparams: tuple
     Additional arguments required by func.
  params: 1D or 2D ndarray
     Set of initial fitting parameters for func.  If 2D, of shape
     (nparams, nchains), it is assumed that it is one set for each chain.
  pmin: 1D ndarray
     Lower boundaries of the posteriors.
  pmax: 1D ndarray
     Upper boundaries of the posteriors.
  stepsize: 1D ndarray
     Proposal jump scale.  If a values is 0, keep the parameter fixed.
     Negative values indicate a shared parameter (See Note 1).
  prior: 1D ndarray
     Parameter prior distribution means (See Note 2).
  priorlow: 1D ndarray
     Lower prior uncertainty values (See Note 2).
  priorup: 1D ndarray
     Upper prior uncertainty values (See Note 2).
  nchains: Scalar
     Number of simultaneous chains to run.
  nproc: Integer
     The number of processors for the MCMC chains (consider that MC3 uses
     one other CPU for the central hub).
  nsamples: Scalar
     Total number of samples.
  walk: String
     Random walk algorithm:
     - 'mrw':  Metropolis random walk.
     - 'demc': Differential Evolution Markov chain.
     - 'snooker': DEMC-z with snooker update.
  wlike: Boolean
     If True, calculate the likelihood in a wavelet-base.  This requires
     three additional parameters (See Note 3).
  leastsq: Boolean
     Perform a least-square minimization before the MCMC run.
  lm: Boolean
     If True use the Levenberg-Marquardt algorithm for the optimization.
     If False, use the Trust Region Reflective algorithm.
  chisqscale: Boolean
     Scale the data uncertainties such that the reduced chi-squared = 1.
  grtest: Boolean
     Run Gelman & Rubin test.
  grbreak: Float
     Gelman-Rubin convergence threshold to stop the MCMC (I'd suggest
     grbreak ~ 1.001--1.005).  Do not break if grbreak=0.0 (default).
  grnmin: Integer or float
     Minimum number of valid samples required for grbreak.
     If grnmin is integer, require at least grnmin samples to break
     out of the MCMC.
     If grnmin is a float (in the range 0.0--1.0), require at least
     grnmin * maximum number of samples to break out of the MCMC.
  burnin: Scalar
     Burned-in (discarded) number of iterations at the beginning
     of the chains.
  thinning: Integer
     Thinning factor of the chains (use every thinning-th iteration) used
     in the GR test and plots.
  fgamma: Float
     Proposals jump scale factor for DEMC's gamma.
     The code computes: gamma = fgamma * 2.38 / sqrt(2*Nfree)
  fepsilon: Float
     Jump scale factor for DEMC's support distribution.
     The code computes: e = fepsilon * Normal(0, stepsize)
  hsize: Integer
     Number of initial samples per chain.
  kickoff: String
     Flag to indicate how to start the chains:
       'normal' for normal distribution around initial guess, or
       'uniform' for uniform distribution withing the given boundaries.
  plots: Bool
     If True plot parameter traces, pairwise-posteriors, and posterior
     histograms.
  ioff: Bool
     If True, set plt.ioff(), i.e., do not display figures on screen.
  showbp: Bool
     If True, show best-fitting values in histogram and pairwise plots.
  savefile: String
     If not None, filename to store allparams and other MCMC results.
  savemodel: String
     If not None, filename to store the values of the evaluated function
     (with np.save).
  resume: Boolean
     If True resume a previous run.
  rms: Boolean
     If True, calculate the RMS of the residuals: data - bestmodel.
  log: String or FILE pointer
     Filename or File object to write log.
  parname: 1D string ndarray
     List of parameter names to display on output figures (including
     fixed and shared).
  full_output:  Bool
     If True, return the full posterior sample, including the burned-in
     iterations.

  Returns
  -------
  bestp: 1D ndarray
     Array of the best-fitting parameters (including fixed and shared).
  CRlo:  1D ndarray
     The lower boundary of the marginal 68%-highest posterior density
     (the credible region) for each parameter, with respect to bestp.
  CRhi:  1D ndarray
     The upper boundary of the marginal 68%-highest posterior density
     (the credible region) for each parameter, with respect to bestp.
  stdp: 1D ndarray
     Array of the best-fitting parameter uncertainties, calculated as the
     standard deviation of the marginalized, thinned, burned-in posterior.
  posterior: 2D float ndarray
     An array of shape (Nfreepars, Nsamples) with the thinned MCMC posterior
     distribution of the fitting parameters (excluding fixed and shared).
     If full_output is True, the posterior includes the burnin samples.
  Zchain: 1D integer ndarray
     Index of the chain for each sample in posterior.  M0 samples have chain
     index of -1.
  chiout: 4-elements tuple
     Tuple containing the best-fit chi-square, reduced chi-square, scale
     factor to enforce redchisq=1, and the Bayesian information
     criterion (BIC).
     Note: Returned only if chireturn=True.

  Notes
  -----
  1.- To set one parameter equal to another, set its stepsize to the
      negative index in params (Starting the count from 1); e.g.: to set
      the second parameter equal to the first one, do: stepsize[1] = -1.
  2.- If any of the fitting parameters has a prior estimate, e.g.,
        param[i] = p0 +up/-low,
      with up and low the 1sigma uncertainties.  This information can be
      considered in the MCMC run by setting:
      prior[i]    = p0
      priorup[i]  = up
      priorlow[i] = low
      All three: prior, priorup, and priorlow must be set and, furthermore,
      priorup and priorlow must be > 0 to be considered as prior.
  3.- FINDME: WAVELET LIKELIHOOD

  Examples
  --------
  >>> # See https://github.com/pcubillos/MCcubed/tree/master/examples

  Uncredited developers
  ---------------------
  Kevin Stevenson (UCF)
  """
  if ioff:
    plt.ioff()

  # Open log file if input is the filename:
  if isinstance(log, str):
    if resume:
      log = open(log, "aw")
    else:
      log = open(log, "w")
    closelog = True
  else:
    closelog = False

  if resume:
    mu.msg(1, "\n\n{:s}\n{:s}  Resuming previous MCMC run.\n\n".
           format(mu.sep, mu.sep), log)

  mu.msg(1, "\n{:s}\n  Multi-Core Markov-Chain Monte Carlo (MC3).\n"
            "  Version {:d}.{:d}.{:d}.\n"
            "  Copyright (c) 2015-2018 Patricio Cubillos and collaborators.\n"
            "  MC3 is open-source software under the MIT license "
            "(see LICENSE).\n{:s}\n\n".
            format(mu.sep, ver.MC3_VER, ver.MC3_MIN, ver.MC3_REV, mu.sep), log)

  # Import the model function:
  if type(func) in [list, tuple, np.ndarray]:
    if len(func) == 3:
      sys.path.append(func[2])
    fmodule = importlib.import_module(func[1])
    func = getattr(fmodule, func[0])
  elif not callable(func):
    mu.error("'func' must be either, a callable, or an iterable (list, "
             "tuple, or ndarray) of strings with the model function, file, "
             "and path names.", log)

  if nproc is None:  # Default to Nproc = Nchains:
    nproc = nchains
  # Cap the number of processors:
  if nproc >= mpr.cpu_count():
    mu.warning("The number of requested CPUs ({:d}) is >= than the number "
      "of available CPUs ({:d}).  Enforced nproc to {:d}.".format(nproc,
             mpr.cpu_count(), mpr.cpu_count()-1), log)
    nproc = mpr.cpu_count() - 1

  nparams = len(params)  # Number of model params
  ndata   = len(data)    # Number of data values
  # Set default uncertainties:
  if uncert is None:
    uncert = np.ones(ndata)

  # Set data and uncert shared-memory objects:
  sm_data   = mpr.Array(ctypes.c_double, data)
  sm_uncert = mpr.Array(ctypes.c_double, uncert)
  # Re-use variables as an ndarray view of the shared-memory object:
  data   = np.ctypeslib.as_array(sm_data.get_obj())
  uncert = np.ctypeslib.as_array(sm_uncert.get_obj())

  # Set default boundaries:
  if pmin is None:
    pmin = np.zeros(nparams) - np.inf
  if pmax is None:
    pmax = np.zeros(nparams) + np.inf
  # Set default stepsize:
  if stepsize is None:
    stepsize = 0.1 * np.abs(params)
  stepsize = np.asarray(stepsize)
  # Set prior parameter indices:
  if (prior is None) or (priorup is None) or (priorlow is None):
    prior   = priorup = priorlow = np.zeros(nparams)  # Zero arrays
  iprior = np.where(priorlow != 0)[0]

  # Check that initial values lie within the boundaries:
  if (np.any(np.asarray(params) < pmin) or
      np.any(np.asarray(params) > pmax) ):
    ihigh = params > pmax
    ilow  = params < pmin
    pout = ""
    for i in np.arange(nparams):
      if ilow[i]:
        pout += "\np{:02d}:  {: 13.6e} < {: 13.6e}".format(i,pmin[i],params[i])
      elif ihigh[i]:
        pout += "\np{:02d}:  {:13s}   {: 13.6e} > {: 13.6e}".format(
                                                     i, "", params[i], pmax[i])

    mu.error("Some initial-guess values are out of bounds:\n"
       "index  pmin            param           pmax\n{:s}{:s}".format(
       "-----  ------------    ------------    ------------", pout), log)

  nfree    = int(np.sum(stepsize > 0))   # Number of free parameters
  ifree    = np.where(stepsize > 0)[0]   # Free   parameter indices
  ishare   = np.where(stepsize < 0)[0]   # Shared parameter indices
  # Number of model parameters (excluding wavelet parameters):
  if wlike:
    mpars  = nparams - 3
  else:
    mpars  = nparams

  # Initial number of samples:
  M0  = hsize * nchains
  # Number of Z samples per chain:
  nZchain = int(np.ceil(nsamples/nchains/thinning))
  # Number of iterations per chain:
  niter  = nZchain * thinning
  # Total number of Z samples (initial + chains):
  Zlen   = M0 + nZchain*nchains

  # Initialize shared-memory free params array:
  sm_freepars = mpr.Array(ctypes.c_double, nchains*nfree)
  freepars    = np.ctypeslib.as_array(sm_freepars.get_obj())
  freepars    = freepars.reshape((nchains, nfree))

  # Get lowest chi-square and best fitting parameters:
  bestchisq = mpr.Value(ctypes.c_double, np.inf)
  sm_bestp  = mpr.Array(ctypes.c_double, np.copy(params))
  bestp     = np.ctypeslib.as_array(sm_bestp.get_obj())
  # There seems to be a strange behavior with np.ctypeslib.as_array()
  # when the argument is a single-element array. In this case, the
  # returned value is a two-dimensional array, instead of 1D. The
  # following line fixes(?) that behavior:
  if np.ndim(bestp) > 1:
    bestp = bestp.flatten()

  if not resume and niter < burnin:
    mu.error("The number of burned-in samples ({:d}) is greater than "
             "the number of iterations per chain ({:d}).".
             format(burnin, niter), log)

  # Check that output path exists:
  if savefile is not None:
    fpath, fname = os.path.split(os.path.realpath(savefile))
    if not os.path.exists(fpath):
      mu.warning("Output folder path: '{:s}' does not exist. "
                 "Creating new folder.".format(fpath), log)
      os.makedirs(fpath)

  # Intermediate steps to run GR test and print progress report:
  intsteps = (nZchain*nchains) / 10
  report = intsteps
  # Initial size of posterior (prior to this MCMC sample):
  size0 = M0

  if resume:
    oldrun = np.load(savefile)
    Zold = oldrun["Z"]
    Zlen_old = np.shape(Zold)[0]  # Previous MCMC
    Zchain_old = oldrun["Zchain"]
    # Redefine Zlen to include the previous runs:
    Zlen = Zlen_old + nZchain*nchains
    size0 = Zlen_old

  # Allocate arrays with variables:
  numaccept = mpr.Value(ctypes.c_int, 0)
  outbounds = mpr.Array(ctypes.c_int, nfree)  # Out of bounds proposals

  #if savemodel is not None:
  #  allmodel = np.zeros((nchains, ndata, niter)) # Fit model

  # Z array with the chains history:
  sm_Z = mpr.Array(ctypes.c_double, Zlen*nfree)
  Z    = np.ctypeslib.as_array(sm_Z.get_obj())
  Z    = Z.reshape((Zlen, nfree))

  # Chi-square value of Z:
  sm_Zchisq = mpr.Array(ctypes.c_double, Zlen)
  Zchisq = np.ctypeslib.as_array(sm_Zchisq.get_obj())
  # Chain index for given state in the Z array:
  sm_Zchain = mpr.Array(ctypes.c_int, -np.ones(Zlen, np.int))
  Zchain = np.ctypeslib.as_array(sm_Zchain.get_obj())
  # Current number of samples in the Z array:
  Zsize  = mpr.Value(ctypes.c_int, M0)
  # Burned samples in the Z array per chain:
  Zburn  = int(burnin/thinning)

  # Include values from previous run:
  if resume:
    Z[0:Zlen_old,:] = Zold
    Zchisq[0:Zlen_old] = oldrun["Zchisq"]
    Zchain[0:Zlen_old] = oldrun["Zchain"]
    # Redefine Zsize:
    Zsize.value = Zlen_old
    numaccept.value = int(oldrun["numaccept"])
  # Set GR N-min:
  if   isinstance(grnmin, int):
    pass
  elif isinstance(grnmin, float):
    grnmin = int(grnmin*(Zlen-M0-Zburn*nchains))
  else:
    mu.error("Invalid grnmin argument.")
  # Add these to compare grnmin to Zsize (which also include them):
  grnmin += int(M0 + Zburn*nchains)

  # Current length of each chain:
  sm_chainsize = mpr.Array(ctypes.c_int, np.tile(hsize, nchains))
  chainsize = np.ctypeslib.as_array(sm_chainsize.get_obj())

  # Number of chains per processor:
  ncpp = np.tile(int(nchains/nproc), nproc)
  ncpp[0:nchains % nproc] += 1

  # Launch Chains:
  pipe   = []
  chains = []
  for i in np.arange(nproc):
    p = mpr.Pipe()
    pipe.append(p[0])
    chains.append(ch.Chain(func, indparams, p[1], data, uncert,
                           params, freepars, stepsize, pmin, pmax,
                           walk, wlike, prior, priorlow, priorup, thinning,
                           fgamma, fepsilon, Z, Zsize, Zchisq, Zchain, M0,
                           numaccept, outbounds, ncpp[i],
                           chainsize, bestp, bestchisq, i, nproc))

  if resume:
    # Set bestp and bestchisq:
    bestp = oldrun["bestp"]
    bestchisq.value = oldrun["bestchisq"]
    for c in np.arange(nchains):
      chainsize[c] = np.sum(Zchain_old==c)
    chifactor = float(oldrun['chifactor'])
    uncert *= chifactor
  else:
    fitpars = np.asarray(params)
    # Least-squares minimization:
    if leastsq:
      fitchisq, fitbestp, dummy, dummy = mf.modelfit(fitpars, func, data,
         uncert, indparams, stepsize, pmin, pmax, prior, priorlow, priorup, lm)
      # Store best-fitting parameters:
      bestp[ifree] = np.copy(fitbestp[ifree])
      # Store minimum chisq:
      bestchisq.value = fitchisq
      mu.msg(1, "Least-squares best-fitting parameters:\n  {:s}\n\n".
                 format(str(fitbestp[ifree])), log, si=2)

    # Populate the M0 initial samples of Z:
    Z[0] = np.clip(bestp[ifree], pmin[ifree], pmax[ifree])
    for j in np.arange(nfree):
      idx = ifree[j]
      if   kickoff == "normal":   # Start with a normal distribution
        vals = np.random.normal(params[idx], stepsize[idx], M0-1)
        # Stay within pmin and pmax boundaries:
        vals[np.where(vals < pmin[idx])] = pmin[idx]
        vals[np.where(vals > pmax[idx])] = pmax[idx]
        Z[1:M0,j] = vals
      elif kickoff == "uniform":  # Start with a uniform distribution
        Z[1:M0,j] = np.random.uniform(pmin[idx], pmax[idx], M0-1)

    # Evaluate models for initial sample of Z:
    for i in np.arange(M0):
      fitpars[ifree] = Z[i]
      # Update shared parameters:
      for s in ishare:
        fitpars[s] = fitpars[-int(stepsize[s])-1]
      Zchisq[i] = chains[0].eval_model(fitpars, ret="chisq")

    # Best-fitting values (so far):
    Zibest = np.argmin(Zchisq[0:M0])
    bestchisq.value = Zchisq[Zibest]
    bestp[ifree] = np.copy(Z[Zibest])

    # FINDME: think what to do with this:
    #models = np.zeros((nchains, ndata))

    # Scale data-uncertainties such that reduced chisq = 1:
    chifactor = 1.0
    if chisqscale:
      chifactor = np.sqrt(bestchisq.value/(ndata-nfree))
      uncert *= chifactor

      # Re-calculate chisq with the new uncertainties:
      for i in np.arange(M0):
        fitpars[ifree] = Z[i]
        for s in ishare:
          fitpars[s] = fitpars[-int(stepsize[s])-1]
        Zchisq[i] = chains[0].eval_model(fitpars, ret="chisq")

      # Re-calculate best-fitting parameters with new uncertainties:
      if leastsq:
        fitchisq, fitbestp, dummy, dummy = mf.modelfit(fitpars, func, data,
              uncert, indparams, stepsize, pmin, pmax, prior, priorlow,
              priorup, lm)
        bestp[ifree] = np.copy(fitbestp[ifree])
        bestchisq.value = fitchisq
        mu.msg(1, "Least-squares best-fitting parameters (rescaled chisq):\n"
                  "  {:s}\n\n".format(str(fitbestp[ifree])), log, si=2)

  # FINDME: do something with models
  #if savemodel is not None:
  #  allmodel[:,:,0] = models

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Start loop:
  print("Yippee Ki Yay Monte Carlo!")
  mu.msg(1, "Start MCMC chains  ({:s})".format(time.ctime()), log)
  for c in np.arange(nproc):
    chains[c].start()
  bit = bool(1)  # Dummy variable to send through pipe for DEMC
  while True:
    # Proposal jump:
    if walk == "demc":
      # Send bit (for DEMC):
      for j in np.arange(nproc):
        pipe[j].send(bit)
      # Receive chi-square (merely for synchronization):
      for j in np.arange(nproc):
        b = pipe[j].recv()

    # Print intermediate info:
    if (Zsize.value-size0 >= report) or (Zsize.value == Zlen):
      report += intsteps
      mu.progressbar((Zsize.value+1.0-size0)/(nZchain*nchains), log)

      mu.msg(1, "Out-of-bound Trials:\n{:s}".
                 format(str(np.asarray(outbounds[:]))), log, width=80)
      mu.msg(1, "Best Parameters: (chisq={:.4f})\n{:s}".
                 format(bestchisq.value, str(bestp[ifree])), log, width=80)

      # Save current results:
      if savefile is not None:
        np.savez(savefile, Z=Z, Zchain=Zchain)
      #if savemodel is not None:
      #  np.save(savemodel, allmodel)

      # Gelman-Rubin statistics:
      if grtest and np.all(chainsize > (Zburn+hsize)):
        psrf = gr.gelmanrubin(Z, Zchain, Zburn)
        mu.msg(1, "Gelman-Rubin statistics for free parameters:\n{:s}".
                   format(str(psrf)), log, width=80)
        if np.all(psrf < 1.01):
          mu.msg(1, "All parameters have converged to within 1% of unity.", log)
        if (grbreak > 0.0 and np.all(psrf < grbreak) and
            Zsize.value > grnmin):
          with Zsize.get_lock():
            Zsize.value = Zlen
          mu.msg(1, "\nAll parameters satisfy the GR convergence threshold "
                    "of {:g}, stopping the MCMC.".format(grbreak), log)
          break
      if Zsize.value == Zlen:
        break

  for j in np.arange(nproc):  # Make sure to terminate the subprocesses
    chains[j].terminate()

  # The models:
  #if savemodel is not None:
  #  modelstack = allmodel[0,:,burnin:]
  #  for c in np.arange(1, nchains):
  #    modelstack = np.hstack((modelstack, allmodel[c, :, burnin:]))

  # Print out Summary:
  mu.msg(1, "\nFin, MCMC Summary:\n------------------", log)
  # Evaluate model for best fitting parameters:
  fitpars = np.asarray(params)
  fitpars[ifree] = np.copy(bestp[ifree])
  for s in ishare:
    fitpars[s] = fitpars[-int(stepsize[s])-1]
  bestmodel = chains[0].eval_model(fitpars)

  # Truncate sample (if necessary):
  Ztotal = M0 + np.sum(Zchain>=0)
  Zchain = Zchain[:Ztotal]
  Zchisq = Zchisq[:Ztotal]
  Z = Z[:Ztotal]

  # Get indices for samples considered in final analysis:
  good = np.zeros(len(Zchain), bool)
  for c in np.arange(nchains):
    good[np.where(Zchain == c)[0][Zburn:]] = True
  # Values accepted for posterior stats:
  posterior = Z[good]
  pchain    = Zchain[good]

  # Sort the posterior by chain:
  zsort = np.lexsort([pchain])
  posterior = posterior[zsort]
  pchain    = pchain   [zsort]

  # Get some stats:
  nsample   = np.sum(Zchain>=0)*thinning  # Total samples run
  nZsample  = len(posterior)  # Valid samples (after thinning and burning)
  ntotal    = nsample
  BIC       = bestchisq.value + nfree*np.log(ndata)
  if ndata > nfree:
    redchisq  = bestchisq.value/(ndata-nfree)
  else:
    redchisq = np.nan
  sdr       = np.std(bestmodel-data)

  fmtlen = len(str(nsample))
  mu.msg(1, "Total number of samples:            {:{}d}".
             format(nsample,  fmtlen), log, 2)
  mu.msg(1, "Number of parallel chains:          {:{}d}".
             format(nchains,  fmtlen), log, 2)
  mu.msg(1, "Average iterations per chain:       {:{}d}".
             format(nsample//nchains, fmtlen), log, 2)
  mu.msg(1, "Burned-in iterations per chain:     {:{}d}".
             format(burnin,   fmtlen), log, 2)
  mu.msg(1, "Thinning factor:                    {:{}d}".
             format(thinning, fmtlen), log, 2)
  mu.msg(1, "MCMC sample size (thinned, burned): {:{}d}".
             format(nZsample, fmtlen), log, 2)
  mu.msg(1, "Acceptance rate:   {:.2f}%\n".
             format(numaccept.value*100.0/nsample), log, 2)

  # Compute the credible region for each parameter:
  CRlo = np.zeros(nparams)
  CRhi = np.zeros(nparams)
  pdf  = []
  xpdf = []
  for i in np.arange(nfree):
    PDF, Xpdf, HPDmin = mu.credregion(posterior[:,i])
    pdf.append(PDF)
    xpdf.append(Xpdf)
    CRlo[ifree[i]] = np.amin(Xpdf[PDF>HPDmin])
    CRhi[ifree[i]] = np.amax(Xpdf[PDF>HPDmin])
  # CR relative to the best-fitting value:
  CRlo[ifree] -= bestp[ifree]
  CRhi[ifree] -= bestp[ifree]

  # Get the mean and standard deviation from the posterior:
  meanp = np.zeros(nparams, np.double) # Parameters mean
  stdp  = np.zeros(nparams, np.double) # Parameter standard deviation
  meanp[ifree] = np.mean(posterior, axis=0)
  stdp [ifree] = np.std(posterior,  axis=0)
  for s in ishare:
    bestp[s] = bestp[-int(stepsize[s])-1]
    meanp[s] = meanp[-int(stepsize[s])-1]
    stdp [s] = stdp [-int(stepsize[s])-1]
    CRlo [s] = CRlo [-int(stepsize[s])-1]
    CRhi [s] = CRhi [-int(stepsize[s])-1]

  mu.msg(1, "\n      Best fit  Lo Cred.Reg.  Hi Cred.Reg.          Mean     Std. dev.      S/N", log, width=80)
  for i in np.arange(nparams):
    snr  = "{:7.1f}".  format(np.abs(bestp[i])/stdp[i])
    mean = "{: 13.6e}".format(meanp[i])
    lo   = "{: 13.6e}".format(CRlo[i])
    hi   = "{: 13.6e}".format(CRhi[i])
    if   i in ifree:  # Free-fitting value
      pass
    elif i in ishare: # Shared value
      snr  = "[sh-p{:02d}]".format(-int(stepsize[i]))
    else:             # Fixed value
      snr  = "[fixed]"
      mean = "{: 13.6e}".format(bestp[i])
    mu.msg(1, "{:14.6e} {:>13s} {:>13s} {:>13s} {:13.6e} {:>8s}".
           format(bestp[i], lo, hi, mean, stdp[i], snr), log, width=80)

  if leastsq and bestchisq.value-fitchisq < -3e-8:
    np.set_printoptions(precision=8)
    mu.warning("MCMC found a better fit than the minimizer:\n"
               "MCMC best-fitting parameters:        (chisq={:.8g})\n{:s}\n"
               "Minimizer best-fitting parameters:   (chisq={:.8g})\n"
               "{:s}".format(bestchisq.value, str(bestp[ifree]),
                             fitchisq,  str(fitbestp[ifree])), log)

  fmtl = len("%.4f"%BIC)  # Length of string formatting
  mu.msg(1, " ", log)
  mu.msg(chisqscale, "sqrt(reduced chi-squared) factor: {:{}.4f}".
                      format(chifactor, fmtl), log, 2)
  mu.msg(1, "Best-parameter's chi-squared:     {:{}.4f}".
             format(bestchisq.value, fmtl), log, 2)
  mu.msg(1, "Bayesian Information Criterion:   {:{}.4f}".
             format(BIC,             fmtl), log, 2)
  mu.msg(1, "Reduced chi-squared:              {:{}.4f}".
             format(redchisq,        fmtl), log, 2)
  mu.msg(1, "Standard deviation of residuals:  {:.6g}\n".format(sdr), log, 2)

  # Save definitive results:
  if savefile is not None:
    np.savez(savefile, bestp=bestp, Z=Z, Zchain=Zchain, Zchisq=Zchisq,
             CRlo=CRlo, CRhi=CRhi, stdp=stdp, meanp=meanp,
             bestchisq=bestchisq.value, redchisq=redchisq, chifactor=chifactor,
             BIC=BIC, sdr=sdr, numaccept=numaccept.value)
  if savemodel is not None:
    np.save(savemodel, allmodel)

  if rms:
    RMS, RMSlo, RMShi, stderr, bs = ta.binrms(bestmodel-data)

  if plots:
    print("Plotting figures.")
    # Extract filename from savefile:
    if savefile is not None:
      if savefile.rfind(".") == -1:
        fname = savefile
      else:
        # Cut out file extention.
        fname = savefile[:savefile.rfind(".")]
    else:
      fname = "MCMC"
    # Include bestp in posterior plots:
    if showbp:
      bestfreepars = bestp[ifree]
    else:
      bestfreepars = None
    # Trace plot:
    if parname is not None:
      parname = np.asarray(parname)[ifree]
    mp.trace(Z, Zchain=Zchain, burnin=Zburn, parname=parname,
             savefile=fname+"_trace.png")
    # Pairwise posteriors:
    mp.pairwise(posterior,  parname=parname, bestp=bestfreepars,
                savefile=fname+"_pairwise.png")
    # Histograms:
    mp.histogram(posterior, parname=parname, savefile=fname+"_posterior.png",
                 percentile=0.683, pdf=pdf, xpdf=xpdf, bestp=bestfreepars)
    # RMS vs bin size:
    if rms:
      mp.RMS(bs, RMS, stderr, RMSlo, RMShi, binstep=len(bs)//500+1,
                                              savefile=fname+"_RMS.png")
    # Sort of guessing that indparams[0] is the X array for data as in y=y(x):
    if (indparams != [] and
        isinstance(indparams[0], (list, tuple, np.ndarray)) and
        np.size(indparams[0]) == ndata):
      try:
        mp.modelfit(data, uncert, indparams[0], bestmodel,
                                              savefile=fname+"_model.png")
      except:
        pass

  # Close the log file if necessary:
  if closelog:
    log.close()

  # Build the output tuple:
  output = bestp, CRlo, CRhi, stdp

  if full_output:
    output += (Z, Zchain)
  else:
    output += (posterior, pchain)

  chiout = (bestchisq.value, redchisq, chifactor, BIC)

  if chireturn:
    output += (chiout,)

  return output
