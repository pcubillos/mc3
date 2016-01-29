#! /usr/bin/env python

# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["mcmc", "parse"]

import sys, os
import argparse, ConfigParser
import numpy as np

from .. import utils as mu
from .  import mcmc as mc


def mcmc(data=None,     uncert=None,     func=None,      indparams=None,
         params=None,   pmin=None,       pmax=None,      stepsize=None,
         prior=None,    priorlow=None,   priorup=None,
         nsamples=None, nchains=None,    walk=None,      wlike=None,
         leastsq=None,  chisqscale=None, grtest=None,    burnin=None,
         thinning=None, hsize=None,      kickoff=None,
         plots=None,    savefile=None,   savemodel=None, resume=None,
         rms=None,      log=None,        cfile=None):
  """
  MCMC driver routine to execute a Markov-chain Monte Carlo run.

  Parameters
  ----------
  data: 1D ndarray or string
     The data array to be fitted or string with the filename where the data
     array is stored (See Note 3).
  uncert: string or 1D ndarray
     uncertainty array of data or string with the filename where the uncert
     array is stored (See Note 3).
  func: Callable or string-iterable
     The callable function that models data as:
        model = func(params, *indparams)
     Or an iterable (list, tuple, or ndarray) of 3 strings:
        (funcname, modulename, path)
     that specify the function name, function module, and module path.
     If the module is already in the python-path scope, path can be omitted.
  indparams: Tuple of 1D ndarrays or string
     Tuple with additional arguments required by func (one argument per tuple
      element) or filename where indparams is stored (See Note 4).
  params: 1D or 2D ndarray or string
     Array of initial fitting parameters for func.  If 2D, of shape
     (nparams, nchains), it is assumed that it is one set for each chain.
     Or string with filename where params is stored (See Note 3).
  pmin: 1D ndarray or string
     Array with lower boundaries of the posteriors or string with filename
     where pmin is stored (See Note 3).
  pmax: 1D ndarray or string
     Array of upper boundaries of the posteriors or string with filename
     where pmax is stored (See Note 3).
  stepsize: 1D ndarray or string
     Array of proposal jump scales or string with filename where stepsize
     array is stored (See Notes 1, 3).
  prior: 1D ndarray or string
     Array of parameter prior distribution means or string with filename
     where the prior array is stored (See Note 2, 3).
  priorlow: 1D ndarray or string
     Array of lower prior uncertainty values or string with filename
     where priorlow is stored (See Note 2, 3).
  priorup: 1D ndarray or string
     Array of upper prior uncertainty values or string with filename
     where priorup is stored (See Note 2, 3).
  nsamples: Scalar
     Total number of MCMC samples.
  nchains: Scalar
     Number of simultaneous chains to run.
  walk: String
     Random walk algorithm:
     - 'mrw':  Metropolis random walk.
     - 'demc': Differential Evolution Markov chain.
  wlike: Boolean
     Calculate the likelihood in a wavelet base.
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
     If True, resume a previous run (load outputs).
  rms: Boolean
     If True, calculate the RMS of data-bestmodel.
  log: String or file pointer
     Filename to write log.
  cfile: String
     Configuration file name.

  Returns
  -------
  posterior: 2D ndarray
     An array of shape (nfree, nsamples) with the MCMC-sampled posterior
     distribution of the fitting parameters.
  Zchain:  1D integer ndarray
     The index of the chain corresponding to each sample.
  bestp:  1D ndarray
     Array of the best fitting parameters.

  Notes
  -----
  1.- If a stepsize value is 0, keep the parameter fixed.
      To set one parameter equal to another, set its stepsize to the
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
  3.- If data, uncert, params, pmin, pmax, stepsize, prior, priorlow,
      or priorup are set as filenames, the file must contain one value per
      line.
      For simplicity, the data file can hold both data and uncert arrays.
      In this case, each line contains one value from each array per line,
      separated by an empty-space character.
      Similarly, params can hold: params, pmin, pmax, stepsize, priorlow,
      and priorup.  The file can hold as few or as many array as long as
      they are provided in that exact order.
  4.- An indparams file works differently, the file will be interpreted
      as a list of arguments, one in each line.  If there is more than one
      element per line (empty-space separated), it will be interpreted as
      an array.

  Examples
  --------
  >>> # See examples in: https://github.com/pcubillos/MCcubed/tree/master/examples
  """

  # Get function arguments into a dictionary:
  args = locals()
  sys.argv = ['ipython']

  try:
    # Parse configuration file to a dictionary:
    if cfile is not None  and  not os.path.isfile(cfile):
      mu.error("Configuration file: '{:s}' not found.".format(cfile))
    if cfile:
      config = ConfigParser.SafeConfigParser()
      config.read([cfile])
      defaults = dict(config.items("MCMC"))
    else:
      defaults = {}

    # Get default values from the command-line-arguments parser:
    parser = parse()
    # Set values from the configuration file (if exists):
    parser.set_defaults(**defaults)
    # Extract values (put into cargs) from the parser object:
    cargs, unknown = parser.parse_known_args()

    # Set undefined argument values with values from config file, or from
    # the defaults:
    for key in args.keys():
      if args[key] is None:
        exec("{:s} = cargs.{:s}".format(key, key))

    # Open a log FILE if requested:
    if   isinstance(log, str):
      log = open(log, "w")
      closelog = True
    else:
      log = None
      closelog = False

    # Handle arguments:
    # Read the model-parameters inputs:
    params = mu.isfile(params, 'params', log, 'ascii', False, notnone=True)
    # Unpack if necessary:
    if len(np.shape(params)) > 1:
      ninfo, ndata = np.shape(params)
      if ninfo == 7:         # The priors
        prior    = params[4]
        priorlow = params[5]
        priorup  = params[6]
      if ninfo >= 4:         # The stepsize
        stepsize = params[3]
      if ninfo >= 2:         # The boundaries
        pmin     = params[1]
        pmax     = params[2]
      params = params[0]     # The initial guess

    # Check for the rest of the arguments if necessary:
    pmin     = mu.isfile(pmin,     'pmin',     log, 'ascii')
    pmax     = mu.isfile(pmax,     'pmax',     log, 'ascii')
    stepsize = mu.isfile(stepsize, 'stepsize', log, 'ascii')
    prior    = mu.isfile(prior,    'prior',    log, 'ascii')
    priorlow = mu.isfile(priorlow, 'priorlow', log, 'ascii')
    priorup  = mu.isfile(priorup,  'priorup',  log, 'ascii')

    # Process the data and uncertainties:
    data = mu.isfile(data,     'data',   log, 'bin', False, notnone=True)
    if len(np.shape(data)) > 1:
      uncert = data[1]
      data   = data[0]
    uncert = mu.isfile(uncert, 'uncert', log, 'bin', notnone=True)

    # Process the independent parameters:
    if indparams != []:
      indparams = mu.isfile(indparams, 'indparams', log, 'bin', False)

    # Use a copy of uncert to avoid overwriting it.
    unc = np.copy(uncert)

    # Call MCMC:
    posterior, Zchain, bestp = mc.mcmc(data, uncert=unc,
                        func=func, indparams=indparams,
                        params=params, pmin=pmin, pmax=pmax, stepsize=stepsize,
                        prior=prior, priorlow=priorlow, priorup=priorup,
                        nsamples=nsamples, nchains=nchains, walk=walk,
                        wlike=wlike,
                        leastsq=leastsq, chisqscale=chisqscale,
                        grtest=grtest, burnin=burnin,
                        thinning=thinning, hsize=hsize, kickoff=kickoff,
                        plots=plots, savefile=savefile, savemodel=savemodel,
                        resume=resume, rms=rms, log=log)

    # Close the log file if it was opened here:
    if closelog:
      log.close()

    return posterior, Zchain, bestp

  except SystemExit:
    return None


def parse():
  """
  MC3 command-line-arguments parser.
  """
  # Parse the config file from the command line:
  parser = argparse.ArgumentParser(description=__doc__, #add_help=False,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

  # Configuration-file option:
  parser.add_argument("-c", "--cfile",
                       help="Configuration file.", metavar="FILE")
  # MCMC Options:
  group = parser.add_argument_group("MCMC General Options")
  group.add_argument("-n", "--nsamples",  dest="nsamples",
                     type=eval,           action="store", default=int(1e5),
                     help="Number of MCMC samples [default: %(default)s]")
  group.add_argument("-x", "--nchains",   dest="nchains",
                     type=int,            action="store", default=10,
                     help="Number of chains [default: %(default)s]")
  group.add_argument("-w", "--walk",      dest="walk",
                     type=str,            action="store", default="demc",
                     choices=('demc', 'mrw'),
                     help="Random walk algorithm [default: %(default)s]")
  group.add_argument(      "--wlike",     dest="wlike",
                     type=eval,           action="store", default=False,
                     help="Calculate the likelihood in a wavelet base "
                          "[default: %(default)s]")
  group.add_argument(      "--leastsq",   dest="leastsq",
                     type=eval,           action="store", default=False,
                     help="Perform a least-square minimization before the "
                          "MCMC run [default: %(default)s]")
  group.add_argument(     "--chisqscale", dest="chisqscale",
                     type=eval,           action="store", default=False,
                     help="Scale the data uncertainties such that the reduced "
                          "chi-squared = 1. [default: %(default)s]")
  group.add_argument("-g", "--grtest",    dest="grtest",
                     type=eval,           action="store", default=False,
                     help="Run Gelman-Rubin test [default: %(default)s]")
  group.add_argument("-b", "--burnin",    dest="burnin",
                     type=eval,           action="store", default=0,
                     help="Number of burn-in iterations (per chain) "
                          "[default: %(default)s]")
  group.add_argument("-t", "--thinning",  dest="thinning",
                     type=int,            action="store",  default=1,
                     help="Chains thinning factor (use every thinning-th "
                          "iteration) for GR test and plots "
                          "[default: %(default)s]")
  group.add_argument("--hsize",           dest="hsize",
                     type=int,            action="store", default=1,
                     help="Number of initial samples per chain "
                          "[default: %(default)s]")
  group.add_argument("--kickoff",         dest="kickoff",
                     type=str,            action="store", default="normal",
                     choices=("normal", "uniform"),
                     help="Chain's starter mode. [default: %(default)s]")
  group.add_argument(      "--plots",     dest="plots",
                     type=eval,           action="store",  default=False,
                     help="If True plot parameter traces, pairwise posteriors, "
                          "and marginal posterior histograms "
                          "[default: %(default)s]")
  group.add_argument("-o", "--save_file", dest="savefile",
                     type=str,            action="store",  default="output.npy",
                     help="Output filename to store the parameter posterior "
                          "distributions  [default: %(default)s]")
  group.add_argument(      "--savemodel", dest="savemodel",
                     type=str,            action="store",  default=None,
                     help="Output filename to store the evaluated models  "
                          "[default: %(default)s]")
  group.add_argument(      "--resume",    dest="resume",
                     type=eval,           action="store",  default=False,
                     help="If True, resume a previous run (load output) "
                          "[default: %(default)s]")
  group.add_argument(      "--rms",       dest="rms",
                     type=eval,           action="store",  default=False,
                     help="If True, calculate the RMS of (data-bestmodel) "
                          "[default: %(default)s]")
  group.add_argument(      "--log",   dest="log",
                     type=str,            action="store", default=None,
                     help="Log file.")
  # Fitting-parameter Options:
  group = parser.add_argument_group("Fitting-function Options")
  group.add_argument("-f", "--func",      dest="func",
                     type=mu.parray,      action="store", default=None,
                     help="List of strings with the function name, module "
                          "name, and path-to-module [required]")
  group.add_argument("-p", "--params",    dest="params",
                     type=mu.parray,      action="store", default=None,
                     help="Filename or list of initial-guess model-fitting "
                          "parameter [required]")
  group.add_argument("-m", "--pmin",      dest="pmin",
                     type=mu.parray,      action="store", default=None,
                     help="Filename or list of parameter lower boundaries "
                          "[default: -inf]")
  group.add_argument("-M", "--pmax",      dest="pmax",
                     type=mu.parray,      action="store", default=None,
                     help="Filename or list of parameter upper boundaries "
                          "[default: +inf]")
  group.add_argument("-s", "--stepsize",  dest="stepsize",
                     type=mu.parray,      action="store", default=None,
                     help="Filename or list with proposal jump scale "
                          "[default: 0.1*params]")
  group.add_argument("-i", "--indparams", dest="indparams",
                     type=mu.parray,      action="store", default=[],
                     help="Filename or list with independent parameters for "
                          "func [default: None]")
  # Data Options:
  group = parser.add_argument_group("Data Options")
  group.add_argument("-d", "--data",    dest="data",
                     type=mu.parray,    action="store", default=None,
                     help="Filename or list of the data being fitted "
                          "[required]")
  group.add_argument("-u", "--uncert",  dest="uncert",
                     type=mu.parray,    action="store", default=None,
                     help="Filemane or list with the data uncertainties "
                          "[default: ones]")
  group.add_argument(     "--prior",    dest="prior",
                     type=mu.parray,    action="store", default=None,
                     help="Filename or list with parameter prior estimates "
                          "[default: %(default)s]")
  group.add_argument(     "--priorlow", dest="priorlow",
                     type=mu.parray,    action="store", default=None,
                     help="Filename or list with prior lower uncertainties "
                          "[default: %(default)s]")
  group.add_argument(     "--priorup",  dest="priorup",
                     type=mu.parray,    action="store", default=None,
                     help="Filename or list with prior upper uncertainties "
                          "[default: %(default)s]")
  return parser

