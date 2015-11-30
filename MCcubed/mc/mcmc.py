#! /usr/bin/env python

# ******************************* START LICENSE *****************************
#
# Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
# model-parameter best-fitting values and Bayesian posterior
# distributions.
#
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington.  Principal developers included graduate students
# Patricio E. Cubillos and Nate B. Lust, and programmer Madison Stemm.
# Statistical advice came from Thomas J. Loredo.
#
# Copyright (C) 2015 University of Central Florida.  All rights reserved.
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
# Patricio Cubillos <pcubillos@fulbrightmail.org>
# Joseph Harrington <jh@physics.ucf.edu>
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

import os, sys, warnings, time
import argparse, ConfigParser
import ctypes
import numpy as np
import multiprocessing as mpr

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/cfuncs/lib')
import chain    as ch
import gelman_rubin as gr
import modelfit as mf
import mcutils  as mu
import mcplots  as mp
import timeavg  as ta

def mcmc(data,         uncert=None,      func=None,     indparams=[],
         params=None,  pmin=None,        pmax=None,     stepsize=None,
         prior=None,   priorlow=None,    priorup=None,
         nsamples=10,  nchains=10,       walk='demc',   wlike=False,
         leastsq=True, chisqscale=False, grtest=True,   burnin=0,
         thinning=1,   hsize=1,          kickoff='normal',
         plots=False,  savefile=None,    savemodel=None, resume=False,
         rms=False,    log=None):
  """
  This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

  Parameters:
  -----------
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
  log: FILE pointer
     File object to write log into.

  Returns:
  --------
  allparams: 2D ndarray
     An array of shape (nfree, nsamples-nchains*burnin) with the MCMC
     posterior distribution of the fitting parameters.
  bestp: 1D ndarray
     Array of the best fitting parameters.

  Notes:
  ------
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

  Examples:
  ---------
  >>> # See examples: https://github.com/pcubillos/MCcubed/tree/master/examples

  Developers:
  -----------
  Kevin Stevenson    UCF  kevin218@knights.ucf.edu
  Patricio Cubillos  UCF  pcubillos@fulbrightmail.org
  """
  # Import the model function:
  if type(func) in [list, tuple, np.ndarray]:
    if func[0] != 'hack':
      if len(func) == 3:
        sys.path.append(func[2])
      exec('from %s import %s as func'%(func[1], func[0]))
  elif not callable(func):
    mu.error("'func' must be either, a callable, or an iterable (list, "
             "tuple, or ndarray) of strings with the model function, file, "
             "and path names.", log)

  nproc = nchains
  nparams = len(params)  # Number of model params
  ndata   = len(data)    # Number of data values
  # Set default uncertainties:
  if uncert is None:
    uncert = np.ones(ndata)

  print("FLAG 010")
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

  print("FLAG 020")
  nfree    = np.sum(stepsize > 0)        # Number of free parameters
  ifree    = np.where(stepsize > 0)[0]   # Free   parameter indices
  ishare   = np.where(stepsize < 0)[0]   # Shared parameter indices
  # Number of model parameters (excluding wavelet parameters):
  if wlike:
    mpars  = nparams - 3
    mstep = normal
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

  # Intermediate steps to run GR test and print progress report:
  intsteps = Zlen / 10
  report   = intsteps

  print("FLAG 030")
  # Allocate arrays with variables:
  numaccept = mpr.Value(ctypes.c_int, 0)
  outbounds = mpr.Array(ctypes.c_int, nfree)  # Out of bounds proposals

  allparams  = np.zeros((nchains, nfree, niter)) # Parameter's record
  if savemodel is not None:
    allmodel = np.zeros((nchains, ndata, niter)) # Fit model

  # Set up the random walks:
  # Normal Distribution for MRW or DEMC:
  normal    = np.random.normal(0, stepsize[ifree], (nchains, niter, nfree))

  print("FLAG 040")
  # Generate indices for the chains such r[c] != c:
  r1 = np.random.randint(0, nchains-1, (nchains, niter))
  r2 = np.random.randint(0, nchains-1, (nchains, niter))
  for c in np.arange(nchains):
    r1[c][np.where(r1[c]==c)] = nchains-1
    r2[c][np.where(r2[c]==c)] = nchains-1

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

  print("The size of Z is {}".format(Zlen*nfree))
  print("The size of Zlen is {}".format(Zlen))
  print("Zsize is {}".format(Zsize.value))

  print("FLAG 050")
  # Uniform random distribution for the Metropolis acceptance rule:
  unif = np.random.uniform(0, 1, (nchains, niter))

  if   walk == "mrw":   # Proposal jumps
    pass
  elif walk == "demc":  # Support random distribution
    pass
  elif walk == "snooker":
    # See: ter Braak & Vrugt (2008), page 439:
    sgamma = np.random.uniform(1.2, 2.2, (nchains, niter))

  # Get lowest chi-square and best fitting parameters:
  bestchisq = mpr.Value(ctypes.c_double, np.inf)
  # FINDME: params
  sm_bestp  = mpr.Array(ctypes.c_double, np.copy(params))
  bestp     = np.ctypeslib.as_array(sm_bestp.get_obj())
  #bestmodel = np.copy(models[np.argmin(chisq)])

  print("FLAG 060")
  timeout = 10.0  # FINDME: set as option
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
                           Z, Zsize, Zlen, Zchisq, Zchain, M0,
                           numaccept, outbounds,
                           normal[i], unif[i], r1[i], r2[i],
                           chainsize, bestp, bestchisq, i, timeout))
    # FINDME: close p[1] ??

  print("FLAG 070")
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
    Zchisq[i] = chains[0].eval_model(fitpars)

  # Best-fitting values (so far):
  Zibest = np.argmin(Zchisq[0:M0])
  bestchisq.value = Zchisq[Zibest]
  bestp[:] = np.copy(Z[Zibest])

  print("FLAG 080")
  # FINDME: Un-break this code
  if resume:
    oldparams = np.load(savefile)
    nold = np.shape(oldparams)[2] # Number of old-run iterations
    allparams = np.dstack((oldparams, allparams))
    if savemodel is not None:
      allmodel  = np.dstack((np.load(savemodel), allmodel))
    # Set params to the last-iteration state of the previous run:
    params = np.repeat(params, nchains, 0)
    params[:,ifree] = oldparams[:,:,-1]
  else:
    nold = 0

  # Least-squares minimization:
  if leastsq:
    fitargs = (params, chains[0].eval_model, data, uncert, [],
               stepsize, pmin, pmax, prior, priorlow, priorup)
    fitchisq, dummy = mf.modelfit(params[ifree], args=fitargs)
    fitbestp = np.copy(params[ifree])
    mu.msg(1, "Least-squares best fitting parameters:\n{:s}\n".
               format(str(fitbestp)), log)

  print("FLAG 090")
  # Calculate chi-squared for model using current params:
  models = np.zeros((nchains, ndata))
  # FINDME: think what to do with this.

  # Scale data-uncertainties such that reduced chisq = 1:
  chifactor = 1.0
  if chisqscale:
    chifactor = np.sqrt(np.amin(chisq)/(ndata-nfree))
    uncert *= chifactor

    # Re-calculate chisq with the new uncertainties:
    for c in np.arange(nchains):
      pipe[i].send(params[i])
    for i in np.arange(nchains):
      chisq[i] = pipe[i].recv()

    if leastsq:
      fitchisq = np.copy(chisq[0])

  # FINDME: do something with models
  if savemodel is not None:
    allmodel[:,:,0] = models

  print("FLAG 100")
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Start loop:
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
      # FINDME: Should I leave this inside the Chain() processes?
      #         I would need to find a way to make them to synchronize.
      # FINDME2: What about not synchronizing?  Just let it take the diff
      #          of whatever current states the chains are?

    # Print intermediate info:
    if (Zsize.value > report) or (Zsize.value == Zlen):
      report += intsteps
      mu.progressbar((Zsize.value+1.0)/Zlen, log)
      #print("Zsize: {}".format(Zsize.value))
      mu.msg(1, "Out-of-bound Trials:\n{:s}".
                           format(np.asarray(outbounds[:])),    log)
      mu.msg(1, "Best Parameters: (chisq={:.4f})\n{:s}".
                           format(bestchisq.value, str(bestp)), log)

      # Gelman-Rubin statistics:
      if grtest and np.all(chainsize > (Zburn+hsize)):
        psrf = gr.convergetest(Z, Zchain, Zburn)
        mu.msg(1, "Gelman-Rubin statistics for free parameters:\n{:s}".
                   format(str(psrf)), log)
        if np.all(psrf < 1.01):
          mu.msg(1, "All parameters have converged to within 1% of unity.", log)
      # Save current results:
      if savefile is not None:
        np.save(savefile, allparams[:,:,0:i+nold])
      if savemodel is not None:
        np.save(savemodel, allmodel[:,:,0:i+nold])
      if report > Zlen:
        print(report, Zlen)
        break
    i += 1

  print("FLAG 101")
  print("Zsize: {}".format(Zsize.value))

  # And the models:
  if savemodel is not None:
    modelstack = allmodel[0,:,burnin:]
    for c in np.arange(1, nchains):
      modelstack = np.hstack((modelstack, allmodel[c, :, burnin:]))

  # Print out Summary:
  mu.msg(1, "\nFin, MCMC Summary:\n------------------", log)
  # Evaluate model for best fitting parameters:
  fitpars = np.asarray(params)
  fitpars[ifree] = np.copy(bestp)
  for s in ishare:
    fitpars[s] = fitpars[-int(stepsize[s])-1]
  bestmodel = chains[0].eval_model(fitpars)

  # Get indices for samples considered in final analysis:
  good = np.zeros(len(Zchain), bool)
  for c in np.arange(nchains):
    good[np.where(Zchain == c)[0][Zburn:]] = True
  # Array with stacked chains:
  allstack = Z[good]

  # Get some stats:
  #nsample   = (niter-burnin)*nchains # This sample
  nsample   = niter*nchains  # This sample
  nZsample  = (nZchain-Zburn) * nchains
  ntotal    = (nold+niter-burnin)*nchains
  BIC       = bestchisq.value + nfree*np.log(ndata)
  redchisq  = bestchisq.value/(ndata-nfree)
  sdr       = np.std(bestmodel-data)

  #fmtlen = len(str(ntotal))
  fmtlen = len(str(nsample))
  mu.msg(1, "Total number of samples:            {:{}d}".
             format(nsample,  fmtlen), log, 2)
  mu.msg(1, "Number of iterations per chain:     {:{}d}".
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

  meanp   = np.mean(allstack, axis=0) # Parameters mean
  uncertp = np.std(allstack,  axis=0) # Parameter standard deviation
  mu.msg(1, "Best-fit params    Uncertainties   Signal/Noise       Sample Mean",
         log, 2)
  for i in np.arange(nfree):
    mu.msg(1, "{: 15.7e}  {: 15.7e}   {:12.2f}   {: 15.7e}".
               format(bestp[ifree][i], uncertp[i],
                      np.abs(bestp[ifree][i])/uncertp[i], meanp[i]), log, 2)

  if leastsq and np.any(np.abs((bestp[ifree]-fitbestp)/fitbestp) > 1e-08):
    np.set_printoptions(precision=8)
    mu.warning("MCMC found a better fit than the minimizer:\n"
               "MCMC best-fitting parameters:        (chisq={:.8g})\n{:s}\n"
               "Minimizer best-fitting parameters:   (chisq={:.8g})\n"
               "{:s}".format(bestchisq.value, str(bestp[ifree]),
                             fitchisq,  str(fitbestp)), log)

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
    print("Plotting figures ...")
    # Extract filename from savefile:
    if savefile is not None:
      if savefile.rfind(".") == -1:
        fname = savefile[savefile.rfind("/")+1:] # Cut out file extention.
      else:
        fname = savefile[savefile.rfind("/")+1:savefile.rfind(".")]
    else:
      fname = "MCMC"
    # Trace plot:
    mp.trace(allstack,     thinning=thinning, savefile=fname+"_trace.png")
    # Pairwise posteriors:
    mp.pairwise(allstack,  thinning=thinning, savefile=fname+"_pairwise.png")
    # Histograms:
    mp.histogram(allstack, thinning=thinning, savefile=fname+"_posterior.png")
    # RMS vs bin size:
    if rms:
      mp.RMS(bs, rms, stderr, rmse, binstep=len(bs)/500+1,
                                              savefile=fname+"_RMS.png")
    if indparams != [] and np.size(indparams[0]) == ndata:
      mp.modelfit(data, uncert, indparams[0], bestmodel,
                                              savefile=fname+"_model.png")

  # Save definitive results:
  if savefile is not None:
    np.save(savefile,  allparams)
  if savemodel is not None:
    np.save(savemodel, allmodel)

  #return Z, Zchain
  return allstack, bestp
