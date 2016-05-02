# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ =["mcmc"]

import os, sys, time
import ctypes
import numpy as np
import multiprocessing as mpr

from .  import gelman_rubin as gr
from .  import chain   as ch

from .. import fit     as mf
from .. import utils   as mu
from .. import plots   as mp
from .. import VERSION as ver

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
import timeavg  as ta


def mcmc(data,         uncert=None,      func=None,     indparams=[],
         params=None,  pmin=None,        pmax=None,     stepsize=None,
         prior=None,   priorlow=None,    priorup=None,
         nsamples=10,  nchains=10,       walk='demc',   wlike=False,
         leastsq=True, chisqscale=False, grtest=True,   burnin=0,
         thinning=1,   hsize=1,          kickoff='normal',
         plots=False,  savefile=None,    savemodel=None, resume=False,
         rms=False,    log=None, full_output=False):
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
  nsamples: Scalar
     Total number of samples.
  nchains: Scalar
     Number of simultaneous chains to run.
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
  chisqscale: Boolean
     Scale the data uncertainties such that the reduced chi-squared = 1.
  grtest: Boolean
     Run Gelman & Rubin test.
  burnin: Scalar
     Burned-in (discarded) number of iterations at the beginning
     of the chains.
  thinning: Integer
     Thinning factor of the chains (use every thinning-th iteration) used
     in the GR test and plots.
  hsize: Integer
     Number of initial samples per chain.
  kickoff: String
     Flag to indicate how to start the chains:
       'normal' for normal distribution around initial guess, or
       'uniform' for uniform distribution withing the given boundaries.
  plots: Boolean
     If True plot parameter traces, pairwise-posteriors, and posterior
     histograms.
  savefile: String
     If not None, filename to store allparams (with np.save).
  savemodel: String
     If not None, filename to store the values of the evaluated function
     (with np.save).
  resume: Boolean
     If True resume a previous run.
  rms: Boolean
     If True, calculate the RMS of the residuals: data - bestmodel.
  log: String or FILE pointer
     Filename or File object to write log.
  full_output:  Bool
     If True, return the full posterior sample, including the burned-in
     iterations.

  Returns
  -------
  bestp: 1D ndarray
     Array of the best-fitting parameters (including fixed and shared).
  uncertp: 1D ndarray
     Array of the best-fitting parameter uncertainties, calculated as the
     standard deviation of the marginalized, thinned, burned-in posterior.
  posterior: 2D float ndarray
     An array of shape (Nfreepars, Nsamples) with the thinned MCMC posterior
     distribution of the fitting parameters (excluding fixed and shared).
     If full_output is True, the posterior includes the burnin samples.
  Zchain: 1D integer ndarray
     Index of the chain for each sample in posterior.  M0 samples have chain
     index of -1.

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
  3.- FINDME WAVELET LIKELIHOOD

  Examples
  --------
  >>> # See https://github.com/pcubillos/MCcubed/tree/master/examples

  Uncredited developers
  ---------------------
  Kevin Stevenson (UCF)
  """

  # Open log file if input is the filename:
  if isinstance(log, str):
    log = open(log, "w")
    closelog = True
  else:
    closelog = False

  mu.msg(1, "\n{:s}\n  Multi-Core Markov-Chain Monte Carlo (MC3).\n"
            "  Version {:d}.{:d}.{:d}.\n"
            "  Copyright (c) 2015-2016 Patricio Cubillos and collaborators.\n"
            "  MC3 is open-source software under the MIT license "
            "(see LICENSE).\n{:s}\n\n".
            format(mu.sep, ver.MC3_VER, ver.MC3_MIN, ver.MC3_REV, mu.sep), log)

  # Import the model function:
  if type(func) in [list, tuple, np.ndarray]:
    if len(func) == 3:
      sys.path.append(func[2])
    exec('from {:s} import {:s} as func'.format(func[1], func[0]))
  elif not callable(func):
    mu.error("'func' must be either, a callable, or an iterable (list, "
             "tuple, or ndarray) of strings with the model function, file, "
             "and path names.", log)

  nproc = nchains
  # Cap the number of processors:
  if nproc >= mpr.cpu_count():
    mu.warning("The number of requested CPUs ({:d}) is >= than the number "
      "of available CPUs ({:d}).  Enforced nproc to {:d}.".format(nproc,
             mpr.cpu_count(), mpr.cpu_count()-1), log)
    nproc = mpr.cpu_count() - 1
    # Re-set number of chains as well:
    nchains = nproc

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
  if np.any(np.asarray(params) < pmin):
    mu.error("One or more of the initial-guess values ({:s}) are smaller "
       "than the minimum boundary ({:s}).".format(str(params), str(pmin)), log)
  if np.any(np.asarray(params) > pmax):
    mu.error("One or more of the initial-guess values ({:s}) are greater "
       "than the maximum boundary ({:s}).".format(str(params), str(pmax)), log)

  nfree    = np.sum(stepsize > 0)        # Number of free parameters
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

  if niter < burnin:
    mu.error("The number of burned-in samples ({:d}) is greater than "
             "the number of iterations per chain ({:d}).".
             format(burnin, niter), log)

  # Intermediate steps to run GR test and print progress report:
  intsteps = Zlen / 10
  report   = intsteps

  # Allocate arrays with variables:
  numaccept = mpr.Value(ctypes.c_int, 0)
  outbounds = mpr.Array(ctypes.c_int, nfree)  # Out of bounds proposals

  if savemodel is not None:
    allmodel = np.zeros((nchains, ndata, niter)) # Fit model

  # Z array with the chains history:
  sm_Z = mpr.Array(ctypes.c_double, Zlen*nfree)
  Z    = np.ctypeslib.as_array(sm_Z.get_obj())
  Z    = Z.reshape((Zlen, nfree))

  # Chi-square value of Z:
  Zchisq = mpr.Array(ctypes.c_double, Zlen)
  # Chain index for given state in the Z array:
  sm_Zchain = mpr.Array(ctypes.c_int, -np.ones(Zlen, np.int))
  Zchain = np.ctypeslib.as_array(sm_Zchain.get_obj())
  # Current number of samples in the Z array:
  Zsize  = mpr.Value(ctypes.c_int, M0)
  # Burned samples in the Z array per chain:
  Zburn  = int(burnin/thinning)

  # Initialize shared-memory free params array:
  sm_freepars = mpr.Array(ctypes.c_double, nchains*nfree)
  freepars    = np.ctypeslib.as_array(sm_freepars.get_obj())
  freepars    = freepars.reshape((nchains, nfree))

  # Get lowest chi-square and best fitting parameters:
  bestchisq = mpr.Value(ctypes.c_double, np.inf)
  sm_bestp  = mpr.Array(ctypes.c_double, np.copy(params))
  bestp     = np.ctypeslib.as_array(sm_bestp.get_obj())
  #bestmodel = np.copy(models[np.argmin(chisq)])

  # Current length of each chain:
  sm_chainsize = mpr.Array(ctypes.c_int, np.zeros(nchains, int)+hsize)
  chainsize = np.ctypeslib.as_array(sm_chainsize.get_obj())

  # Launch Chains:
  pipe   = []
  chains = []
  for i in np.arange(nproc):
    p = mpr.Pipe()
    pipe.append(p[0])
    chains.append(ch.Chain(func, indparams, p[1], data, uncert,
                           params, freepars, stepsize, pmin, pmax,
                           walk, wlike, prior, priorlow, priorup, thinning,
                           Z, Zsize, Zchisq, Zchain, M0,
                           numaccept, outbounds,
                           chainsize, bestp, bestchisq, i))

  # Populate the M0 initial samples of Z:
  for j in np.arange(nfree):
    idx = ifree[j]
    if   kickoff == "normal":   # Start with a normal distribution
      vals = np.random.normal(params[idx], stepsize[idx], M0)
      # Stay within pmin and pmax boundaries:
      vals[np.where(vals < pmin[idx])] = pmin[idx]
      vals[np.where(vals > pmax[idx])] = pmax[idx]
      Z[0:M0,j] = vals
    elif kickoff == "uniform":  # Start with a uniform distribution
      Z[0:M0,j] = np.random.uniform(pmin[idx], pmax[idx], M0)

  # Evaluate models for initial sample of Z:
  fitpars = np.asarray(params)
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

  # FINDME: Un-break this code
  if resume:
    oldparams = np.load(savefile)
    nold = np.shape(oldparams)[2] # Number of old-run iterations
    allparams = np.dstack((oldparams, allparams))  # FINDME fix
    if savemodel is not None:
      allmodel  = np.dstack((np.load(savemodel), allmodel))
    # Set params to the last-iteration state of the previous run:
    params = np.repeat(params, nchains, 0)
    params[:,ifree] = oldparams[:,:,-1]
  else:
    nold = 0

  # Least-squares minimization:
  if leastsq:
    fitchisq, fitbestp, dummy, dummy = mf.modelfit(fitpars, func,
       data, uncert, indparams, stepsize, pmin, pmax, prior, priorlow, priorup)
    # Store best-fitting parameters:
    bestp[ifree] = np.copy(fitbestp[ifree])
    # Store minimum chisq:
    bestchisq.value = fitchisq
    mu.msg(1, "Least-squares best-fitting parameters:\n  {:s}\n\n".
               format(str(fitbestp[ifree])), log, si=2)

  # FINDME: think what to do with this:
  models = np.zeros((nchains, ndata))

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
      fitchisq, fitbp, dummy, dummy = mf.modelfit(fitpars, func,
        data, uncert, indparams, stepsize, pmin, pmax, prior, priorlow, priorup)
      bestp[ifree] = np.copy(fitbestp[ifree])
      bestchisq.value = fitchisq
      mu.msg(1, "Least-squares best-fitting parameters (rescaled chisq):\n"
                "  {:s}\n\n".format(str(fitbestp[ifree])), log, si=2)

  # FINDME: do something with models
  if savemodel is not None:
    allmodel[:,:,0] = models

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Start loop:
  print("Yippee Ki Yay Monte Carlo!")
  mu.msg(1, "Start MCMC chains  ({:s})".format(time.ctime()), log)
  for c in np.arange(nchains):
    chains[c].start()
  i = 0
  bit = bool(1)  # Dummy variable to send through pipe
  while True:
    # Proposal jump:
    if walk == "demc":
      # Send jump (for DEMC):
      for j in np.arange(nchains):
        pipe[j].send(bit)
      # Receive chi-square (merely for synchronization):
      for j in np.arange(nchains):
        b = pipe[j].recv()

    # Print intermediate info:
    if (Zsize.value > report) or (Zsize.value == Zlen):
      report += intsteps
      mu.progressbar((Zsize.value+1.0)/Zlen, log)
      mu.msg(1, "Out-of-bound Trials:\n{:s}".
                           format(np.asarray(outbounds[:])),    log)
      mu.msg(1, "Best Parameters: (chisq={:.4f})\n{:s}".
                           format(bestchisq.value, str(bestp[ifree])), log)

      # Gelman-Rubin statistics:
      if grtest and np.all(chainsize > (Zburn+hsize)):
        psrf = gr.gelmanrubin(Z, Zchain, Zburn)
        mu.msg(1, "Gelman-Rubin statistics for free parameters:\n{:s}".
                   format(str(psrf)), log)
        if np.all(psrf < 1.01):
          mu.msg(1, "All parameters have converged to within 1% of unity.", log)
      # Save current results:
      if savefile is not None:
        np.savez(savefile, Z=Z, Zchain=Zchain)
      if savemodel is not None:
        np.save(savemodel, allmodel[:,:,0:i+nold])
      if report > Zlen:
        break
    i += 1


  # The models:
  if savemodel is not None:
    modelstack = allmodel[0,:,burnin:]
    for c in np.arange(1, nchains):
      modelstack = np.hstack((modelstack, allmodel[c, :, burnin:]))

  # Print out Summary:
  mu.msg(1, "\nFin, MCMC Summary:\n------------------", log)
  # Evaluate model for best fitting parameters:
  fitpars = np.asarray(params)
  fitpars[ifree] = np.copy(bestp[ifree])
  for s in ishare:
    fitpars[s] = fitpars[-int(stepsize[s])-1]
  bestmodel = chains[0].eval_model(fitpars)

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
  nsample   = niter*nchains  # This sample
  nZsample  = len(posterior)
  ntotal    = nold + nsample
  BIC       = bestchisq.value + nfree*np.log(ndata)
  redchisq  = bestchisq.value/(ndata-nfree)
  sdr       = np.std(bestmodel-data)

  #fmtlen = len(str(ntotal))
  fmtlen = len(str(nsample))
  mu.msg(1, "Total number of samples:            {:{}d}".
             format(nsample,  fmtlen), log, 2)
  mu.msg(1, "Number of parallel chains:          {:{}d}".
             format(nchains,  fmtlen), log, 2)
  mu.msg(1, "Average iterations per chain:       {:{}d}".
             format(niter,    fmtlen), log, 2)
  mu.msg(1, "Burned in iterations per chain:     {:{}d}".
             format(burnin,   fmtlen), log, 2)
  mu.msg(1, "Thinning factor:                    {:{}d}".
             format(thinning, fmtlen), log, 2)
  mu.msg(1, "MCMC sample (thinned, burned) size: {:{}d}".
             format(nZsample, fmtlen), log, 2)
  mu.msg(resume, "Total MCMC sample size:             {:{}d}".
             format(ntotal,   fmtlen), log, 2)
  mu.msg(1, "Acceptance rate:   {:.2f}%\n".
             format(numaccept.value*100.0/nsample), log, 2)

  # Get the mean and standard deviation from the posterior:
  meanp   = np.zeros(nparams, np.double) # Parameter standard deviation
  uncertp = np.zeros(nparams, np.double) # Parameters mean
  meanp  [ifree] = np.mean(posterior, axis=0)
  uncertp[ifree] = np.std(posterior,  axis=0)
  for s in ishare:
    bestp  [s] = bestp  [-int(stepsize[s])-1]
    meanp  [s] = meanp  [-int(stepsize[s])-1]
    uncertp[s] = uncertp[-int(stepsize[s])-1]

  mu.msg(1, "\nBest-fit params   Uncertainties        S/N      Sample "
            "Mean   Note", log, 2)
  for i in np.arange(nparams):
    snr  = "{:8.2f}".  format(np.abs(bestp[i])/uncertp[i])
    mean = "{: 14.7e}".format(meanp[i])
    if i in ifree:  # Free-fitting value
      note = ""
    elif i in ishare: # Shared value
      note = "Shared"
    else:             # Fixed value
      note = "Fixed"
      snr  = "---"
      mean = "---"
    mu.msg(1, "{: 15.7e}   {:13.7e}   {:>8s}   {:>14s}   {:s}".
               format(bestp[i], uncertp[i], snr, mean, note), log, 2)

  if leastsq and np.any(np.abs((bestp-fitbestp)/fitbestp) > 1e-08):
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

  if rms:
    rms, rmse, stderr, bs = ta.binrms(bestmodel-data)

  if plots:
    print("Plotting figures.")
    # Extract filename from savefile:
    if savefile is not None:
      if savefile.rfind(".") == -1:
        fname = savefile[savefile.rfind("/")+1:] # Cut out file extention.
      else:
        fname = savefile[savefile.rfind("/")+1:savefile.rfind(".")]
    else:
      fname = "MCMC"
    # Trace plot:
    mp.trace(Z, Zchain=Zchain, burnin=Zburn, savefile=fname+"_trace.png")
    # Pairwise posteriors:
    mp.pairwise(posterior, savefile=fname+"_pairwise.png")
    # Histograms:
    mp.histogram(posterior, savefile=fname+"_posterior.png")
    # RMS vs bin size:
    if rms:
      mp.RMS(bs, rms, stderr, rmse, binstep=len(bs)/500+1,
                                              savefile=fname+"_RMS.png")
    if indparams != [] and np.size(indparams[0]) == ndata:
      mp.modelfit(data, uncert, indparams[0], bestmodel,
                                              savefile=fname+"_model.png")

  # Save definitive results:
  if savefile is not None:
    np.savez(savefile, bestp=bestp, Z=Z, Zchain=Zchain)
  if savemodel is not None:
    np.save(savemodel, allmodel)

  # Close the log file if necessary:
  if closelog:
    log.close()

  if full_output:
    return bestp, uncertp, Z, Zchain
  else:
    return bestp, uncertp, posterior, pchain
