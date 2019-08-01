# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["mcmc"]

import os
import sys
import time
import importlib
import ctypes
import multiprocessing as mpr
from datetime import date

if sys.version_info.major == 2:
    range = xrange

import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

from .fit_driver import fit
from . import chain   as ch
from . import utils   as mu
from . import stats   as ms
from . import plots   as mp
from .VERSION import __version__


@mu.ignore_system_exit
def mcmc(data=None,     uncert=None,    func=None,      params=None,
         indparams=[],  pmin=None,      pmax=None,      pstep=None,
         prior=None,    priorlow=None,  priorup=None,
         nchains=7,     ncpu=None,      nsamples=None,   sampler=None,
         wlike=False,   leastsq=None,   chisqscale=False,
         grtest=True,   grbreak=0.0,    grnmin=0.5,
         burnin=0,      thinning=1,
         fgamma=1.0,    fepsilon=0.0,
         hsize=10,      kickoff='normal',
         plots=False,   ioff=False,     showbp=True,
         savefile=None, resume=False,
         rms=False,     log=None,       pnames=None,    texnames=None,
         # Deprecated:
         parname=None, nproc=None, stepsize=None,
         full_output=None, chireturn=None, lm=None,
         walk=None):
  """
  This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

  Parameters
  ----------
  data: 1D float ndarray or string
      Data to be fit by func.  If string, path to file containing data.
  uncert: 1D float ndarray
      Uncertainties of data.
  func: Callable or string-iterable
      The callable function that models data as:
          model = func(params, *indparams)
      Or an iterable of 3 strings (funcname, modulename, path)
      that specifies the function name, function module, and module path.
      If the module is already in the python-path scope, path can be omitted.
  params: 1D/2D float ndarray or string
      Set of initial fitting parameters for func.  If 2D, of shape
      (nparams, nchains), it is assumed that it is one set for each chain.
      If string, path to file containing data.
  indparams: tuple or string
      Additional arguments required by func.  If string, path to file
      containing indparams.
  pmin: 1D ndarray
      Lower boundaries for the posterior exploration.
  pmax: 1D ndarray
      Upper boundaries for the posterior exploration.
  pstep: 1D ndarray
      Parameter stepping.  If a value is 0, keep the parameter fixed.
      Negative values indicate a shared parameter (See Note 1).
  prior: 1D ndarray
      Parameter prior distribution means (See Note 2).
  priorlow: 1D ndarray
      Lower prior uncertainty values (See Note 2).
  priorup: 1D ndarray
      Upper prior uncertainty values (See Note 2).
  nchains: Scalar
      Number of simultaneous chains to run.
  ncpu: Integer
      Number of processors for the MCMC chains (MC3 defaults to
      one CPU for each chain plus a CPU for the central hub).
  nsamples: Scalar
      Total number of samples.
  sampler: String
      Sampler algorithm:
      - 'mrw':  Metropolis random walk.
      - 'demc': Differential Evolution Markov chain.
      - 'snooker': DEMC-z with snooker update.
  wlike: Bool
      If True, calculate the likelihood in a wavelet-base.  This requires
      three additional parameters (See Note 3).
  leastsq: String
      If not None, perform a least-square optimization before the MCMC run.
      Select from:
          'lm':  Levenberg-Marquardt (most efficient, but does not obey bounds)
          'trf': Trust Region Reflective
  chisqscale: Boolean
      Scale the data uncertainties such that the reduced chi-squared = 1.
  grtest: Boolean
      Run Gelman & Rubin test.
  grbreak: Float
      Gelman-Rubin convergence threshold to stop the MCMC (I'd suggest
      grbreak ~ 1.001--1.005).  Do not break if grbreak=0.0 (default).
  grnmin: Integer or float
      Minimum number of samples required for grbreak to stop the MCMC.
      If grnmin > 1: grnmin sets the minimum required number of samples.
      If 0 < grnmin < 1: grnmin sets the minimum required nsamples fraction.
  burnin: Integer
      Number of burned-in (discarded) number of iterations at the beginning
      of the chains.
  thinning: Integer
      Thinning factor of the chains (use every thinning-th iteration) used
      in the GR test and plots.
  fgamma: Float
      Proposals jump scale factor for DEMC's gamma.
      The code computes: gamma = fgamma * 2.38 / sqrt(2*Nfree)
  fepsilon: Float
      Jump scale factor for DEMC's support distribution.
      The code computes: e = fepsilon * Normal(0, pstep)
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
  resume: Boolean
      If True resume a previous run.
  rms: Boolean
      If True, calculate the RMS of the residuals: data - best_model.
  log: String or FILE pointer
      Filename or File object to write log.
  pnames: 1D string iterable
      List of parameter names (including fixed and shared parameters)
      to display on output screen and figures.  See also texnames.
      Screen output trims up to the 11th character.
      If not defined, default to texnames.
  texnames: 1D string iterable
      Parameter names for figures, which may use latex syntax.
      If not defined, default to pnames.
  parname: 1D string ndarray
      Deprecated, use pnames instead.
  nproc: Integer
      Deprecated, use ncpu instead.
  stepsize: 1D ndarray
      Deprecated, use pstep instead.
  chireturn:
      Deprecated.
  full_output:  Bool
      Deprecated.
  lm: Bool
      Deprecated, see leastsq.
  walk: String
      Deprecated, use sampler instead.

  Returns
  -------
  mc3_output: Dict
      A Dictionary containing the MCMC posterior distribution and related
      stats, including:
      - Z: thinned posterior distribution of shape [nsamples, nfree].
      - Zchain: chain indices for each sample in Z.
      - Zchisq: chi^2 value for each sample in Z.
      - Zmask: indices that turn Z into the desired posterior (remove burn-in).
      - burnin: number of burned-in samples per chain.
      - meanp: mean of the marginal posteriors.
      - stdp: standard deviation of the marginal posteriors.
      - CRlo: lower boundary of the marginal 68%-highest posterior
            density (the credible region).
      - CRhi: upper boundary of the marginal 68%-HPD.
      - bestp: model parameters for the lowest-chi^2 sample.
      - best_model: model evaluated at bestp.
      - best_chisq: lowest-chi^2 in the sample.
      - red_chisq: reduced chi-squared: chi^2/(Ndata}-Nfree) for the
            best-fitting sample.
      - BIC: Bayesian Information Criterion: chi^2-Nfree log(Ndata)
            for the best-fitting sample.
      - chisq_factor: Uncertainties scale factor to enforce chi^2_red = 1.
      - stddev_residuals: standard deviation of the residuals.
      - acceptance_rate: sample's acceptance rate.

  Notes
  -----
  1.- To set one parameter equal to another, set its pstep to the
      negative index in params (Starting the count from 1); e.g.: to set
      the second parameter equal to the first one, do: pstep[1] = -1.
  2.- If any of the fitting parameters has a prior estimate, e.g.,
        param[i] = p0 +up/-low,
      with up and low the 1sigma uncertainties.  This information can be
      considered in the MCMC run by setting:
      prior[i]    = p0
      priorup[i]  = up
      priorlow[i] = low
      All three: prior, priorup, and priorlow must be set and, furthermore,
      priorup and priorlow must be > 0 to be considered as prior.
  3.- If data, uncert, params, pmin, pmax, pstep, prior, priorlow,
      or priorup are set as filenames, the file must contain one value per
      line.
      For simplicity, the data file can hold both data and uncert arrays.
      In this case, each line contains one value from each array per line,
      separated by an empty-space character.
      Similarly, params can hold: params, pmin, pmax, pstep, priorlow,
      and priorup.  The file can hold as few or as many array as long as
      they are provided in that exact order.
  4.- An indparams file works differently, the file will be interpreted
      as a list of arguments, one in each line.  If there is more than one
      element per line (empty-space separated), it will be interpreted as
      an array.
  5.- FINDME: WAVELET LIKELIHOOD

  Examples
  --------
  >>> # See https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html
  """
  # Logging object:
  if isinstance(log, str):
      log = mu.Log(log, append=resume)
      closelog = True
  else:
      closelog = False
      if log is None:
          log = mu.Log(logname=None)

  log.msg("\n{:s}\n"
      "  Multi-core Markov-chain Monte Carlo (MC3).\n"
      "  Version {}.\n"
      "  Copyright (c) 2015-{:d} Patricio Cubillos and collaborators.\n"
      "  MC3 is open-source software under the MIT license (see LICENSE).\n"
      "{:s}\n\n".format(log.sep, __version__, date.today().year, log.sep))

  # Deprecation warnings (to be removed not before summer 2020):
  if parname is not None:
      log.warning("parname argument is deprecated. Use pnames instead.")
      if pnames is None:
          pnames = parname
  if nproc is not None:
      log.warning("nproc argument is deprecated. Use ncpu instead.")
      if ncpu is None:
          ncpu = nproc
  if stepsize is not None:
      log.warning("stepsize argument is deprecated. Use pstep instead.")
      if pstep is None:
          pstep = stepsize
  if walk is not None:
      log.warning("walk argument is deprecated. Use sampler instead.")
      if sampler is None:
          sampler = walk
  if chireturn is not None:
      log.warning("chireturn argument is deprecated.")
  if full_output is not None:
      log.warning("full_output argument is deprecated.")

  if isinstance(leastsq, bool):
      if leastsq is True:
          leastsq = 'trf' if lm is False else 'lm'
      elif leastsq is False:
          leastsq = None
      log.warning("leastsq as boolean is deprecated.  See docs for new "
          "usage.  Set leastsq={}".format(repr(leastsq)))
  if isinstance(lm, bool):
      log.warning('lm argument is deprecated.  See new usage of leastsq.  '
          'Set leastsq={}'.format(repr(leastsq)))

  if sampler is None:
      log.error("'sampler' is a required argument.")
  if nsamples is None and sampler in ['MRW', 'DEMC', 'snooker']:
      log.error("'nsamples' is a required argument for MCMC runs.")
  if leastsq not in [None, 'lm', 'trf']:
      log.error("Invalid 'leastsq' input ({}). Must select from "
                "['lm', 'trf'].".format(leastsq))

  # Read the model parameters:
  params = mu.isfile(params, 'params', log, 'ascii', False, not_none=True)
  # Unpack if necessary:
  if np.ndim(params) > 1:
      ninfo, ndata = np.shape(params)
      if ninfo == 7:         # The priors
          prior    = params[4]
          priorlow = params[5]
          priorup  = params[6]
      if ninfo >= 4:         # The stepsize
          pstep    = params[3]
      if ninfo >= 2:         # The boundaries
          pmin     = params[1]
          pmax     = params[2]
      else:
          log.error('Invalid format/shape for params input file.')
      params = params[0]     # The initial guess

  # Process data and uncertainties:
  data = mu.isfile(data, 'data', log, 'bin', False, not_none=True)
  if np.ndim(data) > 1:
      data, uncert = data
  # Make local 'uncert' a copy, to avoid overwriting:
  if uncert is None:
      log.error("'uncert' is a required argument.")
  uncert = np.copy(uncert)

  # Process the independent parameters:
  if indparams != []:
      indparams = mu.isfile(indparams, 'indparams', log, 'bin', unpack=False)

  if ioff:
      plt.ioff()

  if resume:
      log.msg("\n\n{:s}\n{:s}  Resuming previous MCMC run.\n\n".
              format(log.sep, log.sep))

  # Import the model function:
  if isinstance(func, (list, tuple, np.ndarray)):
      if len(func) == 3:
          sys.path.append(func[2])
      else:
          sys.path.append(os.getcwd())
      fmodule = importlib.import_module(func[1])
      func = getattr(fmodule, func[0])
  elif not callable(func):
      log.error("'func' must be either a callable or an iterable of strings "
                "with the model function, file, and path names.")

  if ncpu is None:  # Default to Nproc = Nchains:
      ncpu = nchains
  # Cap the number of processors:
  if ncpu >= mpr.cpu_count():
      log.warning("The number of requested CPUs ({:d}) is >= than the number "
                  "of available CPUs ({:d}).  Enforced ncpu to {:d}.".
                 format(ncpu, mpr.cpu_count(), mpr.cpu_count()-1))
      ncpu = mpr.cpu_count() - 1

  nparams = len(params)
  ndata   = len(data)

  # Setup array of parameter names:
  if   pnames is None     and texnames is not None:
      pnames = texnames
  elif pnames is not None and texnames is None:
      texnames = pnames
  elif pnames is None     and texnames is None:
      pnames = texnames = mu.default_parnames(nparams)
  pnames   = np.asarray(pnames)
  texnames = np.asarray(texnames)

  # Set uncert as shared-memory object:
  sm_uncert = mpr.Array(ctypes.c_double, uncert)
  uncert = np.ctypeslib.as_array(sm_uncert.get_obj())

  # Set default boundaries:
  if pmin is None:
      pmin = np.tile(-np.inf, nparams)
  if pmax is None:
      pmax = np.tile( np.inf, nparams)
  pmin = np.asarray(pmin)
  pmax = np.asarray(pmax)
  # Set default pstep:
  if pstep is None:
      pstep = 0.1 * np.abs(params)
  pstep = np.asarray(pstep)
  # Set prior parameter indices:
  if prior is None or priorup is None or priorlow is None:
      prior = priorup = priorlow = np.zeros(nparams)

  # Check that initial values lie within the boundaries:
  if (np.any(np.asarray(params) < pmin)
   or np.any(np.asarray(params) > pmax)):
      pout = ""
      for pname, par, minp, maxp in zip(pnames, params, pmin, pmax):
          if par < minp:
              pout += "\n{:11s}  {: 12.5e} < {: 12.5e}".format(
                  pname[:11], minp, par)
          if par > maxp:
              pout += "\n{:26s}  {: 12.5e} > {: 12.5e}".format(
                  pname[:11], par, maxp)

      log.error("Some initial-guess values are out of bounds:\n"
                "Param name           pmin          value           pmax\n"
                "-----------  ------------   ------------   ------------"
                "{:s}".format(pout))

  nfree  = int(np.sum(pstep > 0))   # Number of free parameters
  ifree  = np.where(pstep > 0)[0]   # Free parameter indices
  ishare = np.where(pstep < 0)[0]   # Shared parameter indices

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
  best_chisq = mpr.Value(ctypes.c_double, np.inf)
  sm_bestp   = mpr.Array(ctypes.c_double, np.copy(params))
  bestp      = np.ctypeslib.as_array(sm_bestp.get_obj())
  # There seems to be a strange behavior with np.ctypeslib.as_array()
  # when the argument is a single-element array. In this case, the
  # returned value is a two-dimensional array, instead of 1D. The
  # following line fixes(?) that behavior:
  if np.ndim(bestp) > 1:
      bestp = bestp.flatten()

  if not resume and niter < burnin:
      log.error("The number of burned-in samples ({:d}) is greater than "
                "the number of iterations per chain ({:d}).".
                 format(burnin, niter))

  # Check that output path exists:
  if savefile is not None:
      fpath, fname = os.path.split(os.path.realpath(savefile))
      if not os.path.exists(fpath):
          log.warning("Output folder path: '{:s}' does not exist. "
                      "Creating new folder.".format(fpath))
          os.makedirs(fpath)

  # Intermediate steps to run GR test and print progress report:
  intsteps = (nZchain*nchains) / 10
  report = intsteps
  # Initial size of posterior (prior to this MCMC sample):
  size0 = M0

  if resume:
      oldrun   = np.load(savefile)
      Zold     = oldrun["Z"]
      Zlen_old = np.shape(Zold)[0]  # Previous MCMC
      Zchain_old = oldrun["Zchain"]
      # Redefine Zlen to include the previous runs:
      Zlen = Zlen_old + nZchain*nchains
      size0 = Zlen_old

  # Allocate arrays with variables:
  numaccept = mpr.Value(ctypes.c_int, 0)
  outbounds = mpr.Array(ctypes.c_int, nfree)  # Out of bounds proposals

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
  if grnmin > 0 and grnmin < 1:  # As a fraction:
      grnmin = int(grnmin*(Zlen-M0-Zburn*nchains))
  elif grnmin > 1:               # As the number of iterations:
      pass
  else:
      log.error("Invalid 'grnmin' argument (minimum number of samples to "
          "stop the MCMC under GR convergence), must either be grnmin > 1"
          "to set the minimum number of samples, or 0 < grnmin < 1"
          "to set the fraction of samples required to evaluate.")
  # Add these to compare grnmin to Zsize (which also include them):
  grnmin += int(M0 + Zburn*nchains)

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
          fgamma, fepsilon, Z, Zsize, Zchisq, Zchain, M0,
          numaccept, outbounds, ncpp[i],
          chainsize, bestp, best_chisq, i, ncpu))

  if resume:
      bestp = oldrun["bestp"]
      best_chisq.value = oldrun["best_chisq"]
      for c in range(nchains):
          chainsize[c] = np.sum(Zchain_old==c)
      chisq_factor = float(oldrun['chisq_factor'])
      uncert *= chisq_factor
  else:
      fitpars = np.asarray(params)
      # Least-squares minimization:
      if leastsq is not None:
          fit_outputs = fit(data, uncert, func, fitpars, indparams,
              pstep, pmin, pmax, prior, priorlow, priorup, leastsq)
          # Store best-fitting parameters:
          bestp[ifree] = np.copy(fit_outputs['bestp'][ifree])
          # Store minimum chisq:
          best_chisq.value = fit_outputs['chisq']
          log.msg("Least-squares best-fitting parameters:\n  {:s}\n\n".
                   format(str(fit_outputs['bestp'])), si=2)

      # Populate the M0 initial samples of Z:
      Z[0] = np.clip(bestp[ifree], pmin[ifree], pmax[ifree])
      for j in range(nfree):
          idx = ifree[j]
          if kickoff == "normal":   # Start with a normal distribution
              vals = np.random.normal(params[idx], pstep[idx], M0-1)
              # Stay within pmin and pmax boundaries:
              vals[np.where(vals < pmin[idx])] = pmin[idx]
              vals[np.where(vals > pmax[idx])] = pmax[idx]
              Z[1:M0,j] = vals
          elif kickoff == "uniform":  # Start with a uniform distribution
              Z[1:M0,j] = np.random.uniform(pmin[idx], pmax[idx], M0-1)

      # Evaluate models for initial sample of Z:
      for i in range(M0):
          fitpars[ifree] = Z[i]
          # Update shared parameters:
          for s in ishare:
              fitpars[s] = fitpars[-int(pstep[s])-1]
          Zchisq[i] = chains[0].eval_model(fitpars, ret="chisq")

      # Best-fitting values (so far):
      Zibest = np.argmin(Zchisq[0:M0])
      best_chisq.value = Zchisq[Zibest]
      bestp[ifree] = np.copy(Z[Zibest])

      # Scale data-uncertainties such that reduced chisq = 1:
      chisq_factor = 1.0
      if chisqscale:
          chisq_factor = np.sqrt(best_chisq.value/(ndata-nfree))
          uncert *= chisq_factor

          # Re-calculate chisq with the new uncertainties:
          for i in range(M0):
              fitpars[ifree] = Z[i]
              for s in ishare:
                  fitpars[s] = fitpars[-int(pstep[s])-1]
              Zchisq[i] = chains[0].eval_model(fitpars, ret="chisq")

          # Re-calculate best-fitting parameters with new uncertainties:
          if leastsq is not None:
              fit_outputs = fit(data, uncert, func, fitpars, indparams,
                  pstep, pmin, pmax, prior, priorlow, priorup, leastsq)
              bestp[ifree] = np.copy(fit_outputs['bestp'][ifree])
              best_chisq.value = fit_outputs['chisq']
              log.msg("Least-squares best-fitting parameters (rescaled chisq):"
                      "\n  {:s}\n\n".format(str(fit_outputs['bestp'])), si=2)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Start loop:
  print("Yippee Ki Yay Monte Carlo!")
  log.msg("Start MCMC chains  ({:s})".format(time.ctime()))
  for chain in chains:
      chain.start()
  bit = bool(1)  # Dummy variable to send through pipe for DEMC
  while True:
      # Proposal jump:
      if sampler == "demc":
          # Send and receive bit for synchronization:
          for pipe in pipes:
              pipe.send(bit)
          for pipe in pipes:
              b = pipe.recv()

      # Print intermediate info:
      if (Zsize.value-size0 >= report) or (Zsize.value == Zlen):
          report += intsteps
          log.progressbar((Zsize.value+1.0-size0)/(nZchain*nchains))

          log.msg("Out-of-bound Trials:\n{:s}".
                  format(str(np.asarray(outbounds[:]))),      width=80)
          log.msg("Best Parameters: (chisq={:.4f})\n{:s}".
                  format(best_chisq.value, str(bestp[ifree])), width=80)

          # Save current results:
          if savefile is not None:
              np.savez(savefile, Z=Z, Zchain=Zchain)

          # Gelman-Rubin statistics:
          if grtest and np.all(chainsize > (Zburn+hsize)):
              psrf = ms.gelman_rubin(Z, Zchain, Zburn)
              log.msg("Gelman-Rubin statistics for free parameters:\n{:s}".
                       format(str(psrf)), width=80)
              if np.all(psrf < 1.01):
                  log.msg("All parameters converged to within 1% of unity.")
              if (grbreak > 0.0 and np.all(psrf < grbreak) and
                  Zsize.value > grnmin):
                  with Zsize.get_lock():
                      Zsize.value = Zlen
                  log.msg("\nAll parameters satisfy the GR convergence "
                      "threshold of {:g}, stopping the MCMC.".format(grbreak))
                  break
          if Zsize.value == Zlen:
              break

  for chain in chains:  # Make sure to terminate the subprocesses
      chain.terminate()

  # Print out Summary:
  log.msg("\nMCMC Summary:\n-------------")
  # Evaluate model for best fitting parameters:
  fitpars = np.asarray(params)
  fitpars[ifree] = np.copy(bestp[ifree])
  for s in ishare:
      fitpars[s] = fitpars[-int(pstep[s])-1]
  best_model = chains[0].eval_model(fitpars)

  # Truncate sample (if necessary):
  Ztotal = M0 + np.sum(Zchain>=0)
  Zchain = Zchain[:Ztotal]
  Zchisq = Zchisq[:Ztotal]
  Z = Z[:Ztotal]

  # And remove burn-in samples:
  posterior, zchain, Zmask = mu.burn(Z=Z, Zchain=Zchain, burnin=Zburn)

  # Get some stats:
  nsample   = np.sum(Zchain>=0)*thinning  # Total samples run
  nZsample  = len(posterior)  # Valid samples (after thinning and burning)
  BIC       = best_chisq.value + nfree*np.log(ndata)
  if ndata > nfree:
      red_chisq  = best_chisq.value/(ndata-nfree)
  else:
      red_chisq = np.nan
  sdr = np.std(best_model-data)

  fmt = len(str(nsample))
  log.msg("Total number of samples:            {:{}d}".
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
          format(nZsample, fmt), indent=2)
  log.msg("Acceptance rate:   {:.2f}%\n".
          format(numaccept.value*100.0/nsample), indent=2)

  # Compute the credible region for each parameter:
  CRlo = np.zeros(nparams)
  CRhi = np.zeros(nparams)
  pdf  = []
  xpdf = []
  for i in range(nfree):
      PDF, Xpdf, HPDmin = ms.cred_region(posterior[:,i])
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
      bestp[s] = bestp[-int(pstep[s])-1]
      meanp[s] = meanp[-int(pstep[s])-1]
      stdp [s] = stdp [-int(pstep[s])-1]
      CRlo [s] = CRlo [-int(pstep[s])-1]
      CRhi [s] = CRhi [-int(pstep[s])-1]

  log.msg("\nParam name     Best fit   Lo HPD CR   Hi HPD CR        Mean    Std dev       S/N"
          "\n----------- ----------------------------------- ---------------------- ---------", width=80)
  for i in range(nparams):
      snr  = "{:.1f}".   format(np.abs(bestp[i])/stdp[i])
      mean = "{: 11.4e}".format(meanp[i])
      lo   = "{: 11.4e}".format(CRlo[i])
      hi   = "{: 11.4e}".format(CRhi[i])
      if   i in ifree:  # Free-fitting value
          pass
      elif i in ishare: # Shared value
          snr = "[share{:02d}]".format(-int(pstep[i]))
      else:             # Fixed value
          snr = "[fixed]"
          mean = "{: 11.4e}".format(bestp[i])
      log.msg("{:<11s} {:11.4e} {:>11s} {:>11s} {:>11s} {:10.4e} {:>9s}".
              format(pnames[i][0:11], bestp[i], lo, hi, mean, stdp[i], snr),
              width=160)

  if leastsq is not None and best_chisq.value-fit_outputs['chisq'] < -3.0e-8:
      np.set_printoptions(precision=8)
      log.warning("MCMC found a better fit than the minimizer:\n"
          "MCMC best-fitting parameters:        (chisq={:.8g})\n{:s}\n"
          "Minimizer best-fitting parameters:   (chisq={:.8g})\n"
          "{:s}".format(best_chisq.value, str(bestp[ifree]),
              fit_outputs['chisq'], str(fit_outputs['bestp'][ifree])))

  fmt = len("{:.4f}".format(BIC))  # Length of string formatting
  log.msg(" ")
  if chisqscale:
      log.msg("sqrt(reduced chi-squared) factor: {:{}.4f}".
              format(chisq_factor, fmt), indent=2)
  log.msg("Best-parameter's chi-squared:     {:{}.4f}".
          format(best_chisq.value, fmt), indent=2)
  log.msg("Bayesian Information Criterion:   {:{}.4f}".
          format(BIC, fmt), indent=2)
  log.msg("Reduced chi-squared:              {:{}.4f}".
          format(red_chisq, fmt), indent=2)
  log.msg("Standard deviation of residuals:  {:.6g}\n".format(sdr), indent=2)

  if savefile is not None or plots or closelog:
      log.msg("\nOutput MCMC files:")

  # Build the output dict:
  output = {
      # The posterior:
      'Z':Z,
      'Zchain':Zchain,
      'Zchisq':Zchisq,
      'Zmask':Zmask,
      'burnin':Zburn,
      # Posterior stats:
      'meanp':meanp,
      'stdp':stdp,
      'CRlo':CRlo,
      'CRhi':CRhi,
      'stddev_residuals':sdr,
      'acceptance_rate':numaccept.value*100.0/nsample,
      # Optimization:
      'bestp':bestp,
      'best_model':best_model,
      'best_chisq':best_chisq.value,
      'red_chisq':red_chisq,
      'chisq_factor':chisq_factor,
      'BIC':BIC,
      }

  # Save definitive results:
  if savefile is not None:
      np.savez(savefile, **output)
      log.msg("'{:s}'".format(savefile), indent=2)

  if rms:
      RMS, RMSlo, RMShi, stderr, bs = ms.time_avg(best_model-data)

  if plots:
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
      bestfreepars = bestp[ifree] if showbp else None
      # Trace plot:
      mp.trace(Z, Zchain=Zchain, burnin=Zburn, pnames=texnames[ifree],
          savefile=fname+"_trace.png")
      log.msg("'{:s}'".format(fname+"_trace.png"), indent=2)
      # Pairwise posteriors:
      mp.pairwise(posterior,  pnames=texnames[ifree], bestp=bestfreepars,
          savefile=fname+"_pairwise.png")
      log.msg("'{:s}'".format(fname+"_pairwise.png"), indent=2)
      # Histograms:
      mp.histogram(posterior, pnames=texnames[ifree], bestp=bestfreepars,
          savefile=fname+"_posterior.png",
          percentile=0.683, pdf=pdf, xpdf=xpdf)
      log.msg("'{:s}'".format(fname+"_posterior.png"), indent=2)
      # RMS vs bin size:
      if rms:
          mp.rms(bs, RMS, stderr, RMSlo, RMShi, binstep=len(bs)//500+1,
                 savefile=fname+"_RMS.png")
          log.msg("'{:s}'".format(fname+"_RMS.png"), indent=2)
      # Sort of guessing that indparams[0] is the X array for data as in y=y(x):
      if (indparams != []
          and isinstance(indparams[0], (list, tuple, np.ndarray))
          and np.size(indparams[0]) == ndata):
          try:
              mp.modelfit(data, uncert, indparams[0], best_model,
                          savefile=fname+"_model.png")
              log.msg("'{:s}'".format(fname+"_model.png"), indent=2)
          except:
              pass

  # Close the log file if necessary:
  if closelog:
      log.msg("'{:s}'".format(log.logname), indent=2)
      log.close()

  return output
