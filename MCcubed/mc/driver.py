# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["mcmc", "parse"]

import sys, os
import argparse
import numpy as np
from io import IOBase

# Python2 and Python3 compatibility:
if sys.version_info.major == 3:
  import configparser
  file = IOBase
else:
  import ConfigParser as configparser

from .. import utils as mu
from .  import mcmc as mc


def mcmc(data=None,     uncert=None,     func=None,       indparams=None,
         params=None,   pmin=None,       pmax=None,       stepsize=None,
         prior=None,    priorlow=None,   priorup=None,
         nproc=None,    nsamples=None,   nchains=None,    walk=None,
         wlike=None,    leastsq=None,    lm=None,         chisqscale=None,
         grtest=None,   grbreak=None,    grnmin=None,
         burnin=None,   thinning=None,
         fgamma=None,   fepsilon=None,   hsize=None,      kickoff=None,
         plots=None,    ioff=None,       showbp=None,
         savefile=None, savemodel=None,  resume=None,
         rms=None,      log=None,        cfile=None,      parname=None,
         full_output=None, chireturn=None):
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
  nproc: Integer
     The number of processors for the MCMC chains (consider that MC3 uses
     one extra CPU for the central hub).
  nsamples: Scalar
     Total number of MCMC samples.
  nchains: Scalar
     Number of simultaneous chains to run.
  walk: String
     Random walk algorithm:
     - 'mrw':  Metropolis random walk.
     - 'demc': Differential Evolution Markov chain.
     - 'snooker': DEMC-z with snooker update.
  wlike: Boolean
     Calculate the likelihood in a wavelet base.
  leastsq: Boolean
     Perform a least-square optimization before the MCMC run.
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
     If not None, filename to store allparams (with np.save).
  savemodel: String
     If not None, filename to store the values of the evaluated function
     (with np.save).
  resume: Boolean
     If True, resume a previous MCMC run.
  rms: Boolean
     If True, calculate the RMS of data-bestmodel.
  log: String or file pointer
     Filename to write log.
  cfile: String
     Configuration file name.
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
  >>> # See https://github.com/pcubillos/MCcubed/tree/master/examples
  """

  # Get function arguments into a dictionary:
  args = dict(locals())
  args.pop("cfile")     # Remove cfile from dict
  sys.argv = ['ipython']

  try:
    # Parse configuration file to a dictionary:
    if cfile is not None  and  not os.path.isfile(cfile):
      mu.error("Configuration file: '{:s}' not found.".format(cfile))
    if cfile:
      config = configparser.SafeConfigParser()
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

    # Put Namespace into a dict so we can extract their values:
    cargs = vars(cargs)
    # Set undefined argument values with values from config file, or from
    # the defaults:
    for key in args.keys():
      if args[key] is None:
        args[key] = cargs[key]

    else:
      # Open a log FILE if requested:
      if   isinstance(args["log"], str):
        if args["resume"]:
          log = args["log"] = open(args["log"], "aw")  # Append
        else:
          log = args["log"] = open(args["log"], "w")   # New file
        closelog = True
      elif isinstance(args["log"], file):
        log = args["log"]
        closelog = False
      else:
        log = args["log"] = None
        closelog = False

    # Handle arguments:
    # Read the model-parameters inputs:
    args["params"] = mu.isfile(args["params"], 'params', log, 'ascii',
                               False, notnone=True)
    # Unpack if necessary:
    if len(np.shape(args["params"])) > 1:
      ninfo, ndata = np.shape(args["params"])
      if ninfo == 7:         # The priors
        args["prior"]    = args["params"][4]
        args["priorlow"] = args["params"][5]
        args["priorup"]  = args["params"][6]
      if ninfo >= 4:         # The stepsize
        args["stepsize"] = args["params"][3]
      if ninfo >= 2:         # The boundaries
        args["pmin"]     = args["params"][1]
        args["pmax"]     = args["params"][2]
      args["params"] = args["params"][0]     # The initial guess

    # Check for the rest of the arguments if necessary:
    args["pmin"]     = mu.isfile(args["pmin"],     'pmin',     log, 'ascii')
    args["pmax"]     = mu.isfile(args["pmax"],     'pmax',     log, 'ascii')
    args["stepsize"] = mu.isfile(args["stepsize"], 'stepsize', log, 'ascii')
    args["prior"]    = mu.isfile(args["prior"],    'prior',    log, 'ascii')
    args["priorlow"] = mu.isfile(args["priorlow"], 'priorlow', log, 'ascii')
    args["priorup"]  = mu.isfile(args["priorup"],  'priorup',  log, 'ascii')

    # Process the data and uncertainties:
    args["data"] = mu.isfile(args["data"], 'data', log, 'bin',
                             False, notnone=True)
    if len(np.shape(args["data"])) > 1:
      args["uncert"] = args["data"][1]
      args["data"]   = args["data"][0]
    args["uncert"] = np.copy(mu.isfile(args["uncert"], 'uncert', log, 'bin',
                               notnone=True))  # To avoid overwriting

    # Process the independent parameters:
    if args["indparams"] != []:
      args["indparams"] = mu.isfile(args["indparams"], 'indparams', log, 'bin',
                                    unpack=False)

    # Call MCMC:
    outputs = mc.mcmc(**args)

    # Close the log file if it was opened here:
    if closelog:
      log.close()

    return outputs

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
  group.add_argument("--nsamples",  dest="nsamples", action="store",
                     type=eval, default=int(1e5),
                     help="Number of MCMC samples [default: %(default)s]")
  group.add_argument("--nchains",   dest="nchains", action="store",
                     type=int,  default=7,
                     help="Number of chains [default: %(default)s]")
  group.add_argument("--nproc",     dest="nproc",   action="store",
                     type=int,  default=None,
                     help="Number of CPUs for the chains [default: nchains+1]")
  group.add_argument("--walk",      dest="walk", action="store",
                     type=str,  default="snooker",
                     help="Random walk algorithm, select from: ['mrw', "
                          "'demc', 'snooker']. [default: %(default)s]")
  group.add_argument("--wlike",     dest="wlike", action="store",
                     type=eval, default=False,
                     help="Calculate the likelihood in a wavelet base "
                          "[default: %(default)s]")
  group.add_argument("--leastsq",   dest="leastsq", action="store",
                     type=eval, default=False,
                     help="Perform a least-square optimiztion before the "
                          "MCMC run [default: %(default)s]")
  group.add_argument("--lm",   dest="lm", action="store",
                     type=eval, default=False,
                     help="Use Levenberg-Marquardt (True) or Trust Region "
                     "Reflective (False) optimization algorithm. "
                     "[default: %(default)s]")
  group.add_argument("--chisqscale", dest="chisqscale", action="store",
                     type=eval, default=False,
                     help="Scale the data uncertainties such that the reduced "
                          "chi-squared = 1. [default: %(default)s]")
  group.add_argument("--grtest",     dest="grtest", action="store",
                     type=eval, default=False,
                     help="Run Gelman-Rubin test [default: %(default)s]")
  group.add_argument("--grbreak",   dest="grbreak", action="store",
                     type=float, default=0.0,
                     help="Gelman-Rubin convergence threshold to stop the "
                          "MCMC.  I'd suggest grbreak ~ 1.001 -- 1.005."
                          "Do not break if grbreak=0.0 (default).")
  group.add_argument("--grnmin",     dest="grnmin", action="store",
                     type=eval, default=0.5, help="Minimum number of valid "
                     "samples required for grbreak.  If grnmin is integer, "
                     "require at least grnmin samples to break out of the "
                     "MCMC. If grnmin is a float (in the range 0.0--1.0), "
                     "require at least grnmin * maximum number of samples to "
                     "break out of the MCMC [default: %(default)s]")
  group.add_argument("--burnin",    dest="burnin", action="store",
                     type=eval, default=0,
                     help="Number of burn-in iterations (per chain) "
                          "[default: %(default)s]")
  group.add_argument("--thinning",  dest="thinning", action="store",
                     type=int,  default=1,
                     help="Chains thinning factor (use every thinning-th "
                          "iteration) for GR test and plots "
                          "[default: %(default)s]")
  group.add_argument("--fgamma",    dest="fgamma",   action="store",
                     help="Scaling factor for DEMC's gamma "
                          "[default: %(default)s]",
                     type=float, default=1.0)
  group.add_argument("--fepsilon",  dest="fepsilon", action="store",
                     help="Scaling factor for DEMC's support distribution "
                          "[default: %(default)s]",
                     type=float, default=0.0)
  group.add_argument("--hsize",     dest="hsize", action="store",
                     type=int,  default=10,
                     help="Number of initial samples per chain "
                          "[default: %(default)s]")
  group.add_argument("--kickoff",   dest="kickoff", action="store",
                     type=str,  default="normal",
                     help="Chain's starter mode, select between: ['normal', "
                          "'uniform']. [default: %(default)s]")
  group.add_argument("--plots",     dest="plots", action="store",
                     type=eval, default=False,
                     help="If True, generate output figures. "
                          "[default: %(default)s]")
  group.add_argument("--ioff",     dest="ioff", action="store",
                     type=eval, default=False,
                     help="If True, set plt.ioff(), i.e., do not display "
                          "figures on screen [default: %(default)s]")
  group.add_argument("--showbp",   dest="showbp", action="store",
                     type=eval, default=True,
                     help="If True, show best-fitting values in histogram "
                          "and pairwise plots [default: %(default)s]")
  group.add_argument("--save_file", dest="savefile", action="store",
                     type=str,  default=None,
                     help="Output npz filename to store the parameter "
                          "posterior distributions [default: %(default)s]")
  group.add_argument("--savemodel", dest="savemodel", action="store",
                     type=str,  default=None,
                     help="Output filename to store the evaluated models  "
                          "[default: %(default)s]")
  group.add_argument("-r", "--resume", dest="resume", action="store_true",
                     default=False,
                     help="If set, resume a previous run (load output).")
  group.add_argument("--rms",       dest="rms", action="store",
                     type=eval, default=False,
                     help="If True, calculate the RMS of (data-bestmodel) "
                          "[default: %(default)s]")
  group.add_argument("--log",       dest="log", action="store",
                     type=str,  default=None,
                     help="Log file.")
  group.add_argument("--parname",   dest="parname", action="store",
                     type=mu.parray, default=None,
                     help="List of parameter names. [default: None]")
  group.add_argument("--full_output", dest="full_output", action="store",
                     type=eval, default=False,
                     help="If True, return the full posterior sample, including"
                          " the burnin iterations [default: %(default)s]")
  group.add_argument("--chireturn", dest="chireturn", action="store",
                     type=eval, default=False,
                     help="If True, return chi-squared, red. chi-squared,"
                          "the chi-squared rescaling factor, and the BIC"
                          " [default: %(default)s]")
  # Fitting-parameter Options:
  group = parser.add_argument_group("Fitting-function Options")
  group.add_argument("--func",       dest="func", action="store",
                     type=mu.parray, default=None,
                     help="List of strings with the function name, module "
                          "name, and path-to-module [required]")
  group.add_argument("--params",     dest="params", action="store",
                     type=mu.parray, default=None,
                     help="Filename or list of initial-guess model-fitting "
                          "parameter [required]")
  group.add_argument("--pmin",       dest="pmin", action="store",
                     type=mu.parray, default=None,
                     help="Filename or list of parameter lower boundaries "
                          "[default: -inf for each parameter]")
  group.add_argument("--pmax",       dest="pmax", action="store",
                     type=mu.parray, default=None,
                     help="Filename or list of parameter upper boundaries "
                          "[default: +inf for each parameter]")
  group.add_argument("--stepsize",   dest="stepsize", action="store",
                     type=mu.parray, default=None,
                     help="Filename or list with proposal jump scale. "
                     "[required].  Additionally, parameters with stepsize=0 "
                     "are fixed, parameters with negative stepsize are "
                     "shared (see documentation).")
  group.add_argument("--indparams",  dest="indparams", action="store",
                     type=mu.parray, default=[],
                     help="Filename or list with independent parameters for "
                          "func [default: None]")
  # Data Options:
  group = parser.add_argument_group("Data Options")
  group.add_argument("--data",     dest="data", action="store",
                     type=mu.parray,    default=None,
                     help="Filename or array of the data being fitted "
                          "[required]")
  group.add_argument("--uncert",   dest="uncert", action="store",
                     type=mu.parray,    default=None,
                     help="Filemane or array with the data uncertainties "
                          "[required]")
  group.add_argument("--prior",    dest="prior", action="store",
                     type=mu.parray,    default=None,
                     help="Filename or array with parameter prior estimates "
                          "[default: %(default)s]")
  group.add_argument("--priorlow", dest="priorlow", action="store",
                     type=mu.parray,    default=None,
                     help="Filename or array with prior lower uncertainties "
                          "[default: %(default)s]")
  group.add_argument("--priorup",  dest="priorup", action="store",
                     type=mu.parray,    default=None,
                     help="Filename or array with prior upper uncertainties "
                          "[default: %(default)s]")
  return parser

