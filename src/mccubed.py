#! /usr/bin/env python

# ******************************* START LICENSE *****************************
# 
# Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
# model-parameter best-fitting values and Bayesian posterior
# distributions.
# 
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington.  Principal developers included graduate student
# Patricio E. Cubillos and programmer Madison Stemm.  Statistical advice
# came from Thomas J. Loredo and Nate B. Lust.
# 
# Copyright (C) 2014-2015 University of Central Florida.  All rights
# reserved.
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
# Joseph Harrington <jh@physics.ucf.edu>
# Patricio Cubillos <pcubillos@fulbrightmail.org>
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

import sys, os
import subprocess
import warnings
import argparse, ConfigParser
import timeit
import numpy as np

import mcmc    as mc
import mcutils as mu
start = timeit.default_timer()

def main():
  """
  Multi-Core Markov-chain Monte Carlo (MC3) top-level MCMC driver.

  Notes:
  ------
  1.- To display the full list of arguments, run from the prompt:
      mccubed.py -h
  2.- The command line overwrites over the config file in case an argument
      is defined twice.

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  2014-05-04  patricio  Added cfile argument for Interpreter support.
  2014-05-26  patricio  Re-engineered the MPI support.
  2014-06-26  patricio  Fixed bug with copy when uncert is None.
  2014-09-14  patricio  Write/read now binary files.
  2014-10-23  patricio  Added support for func hack.
  2015-02-04  patricio  Added resume argument.
  2015-04-25  patricio  Re-worked as a driver of mcmc.
  """

  parser = parse()

  # Parse command-line args (right now, just interested in the config file):
  args, unknown = parser.parse_known_args()

  # Parse configuration file to a dictionary:
  if args.cfile is not None and not os.path.isfile(args.cfile):
    mu.error("Configuration file: '{:s}' not found.".format(args.cfile))
  if args.cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([args.cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}
  # Set defaults from the configuration-file values:
  parser.set_defaults(**defaults)
  # Overwrite defaults with the command-line arguments:
  args, unknown = parser.parse_known_args()

  # Unpack configuration-file/command-line arguments:
  cfile      = args.cfile
  nsamples   = args.nsamples
  nchains    = args.nchains
  walk       = args.walk
  wlike      = args.wlike
  leastsq    = args.leastsq
  chisqscale = args.chisqscale
  grtest     = args.grtest
  burnin     = args.burnin
  thinning   = args.thinning
  plots      = args.plots
  savefile   = args.savefile
  savemodel  = args.savemodel
  resume     = args.resume
  rms        = args.rms
  tracktime  = args.tractime

  func      = args.func
  params    = args.params
  pmin      = args.pmin
  pmax      = args.pmax
  stepsize  = args.stepsize
  indparams = args.indparams

  data     = args.data
  uncert   = args.uncert
  prior    = args.prior
  priorup  = args.priorup
  priorlow = args.priorlow
  nprocs   = nchains

  if tracktime:
    start = timeit.default_timer()

  # Call MCMC driver:
  output = mcmc(data, uncert, func, indparams,
                params, pmin, pmax, stepsize,
                prior, priorlow, priorup,
                nsamples, nchains, walk, wlike,
                leastsq, chisqscale, grtest, burnin,
                thinning, plots, savefile, savemodel, resume, rms)

  if tracktime:
    stop = timeit.default_timer()

  if tracktime:
    print("Total execution time: {:.6f} sec".format(stop - start))


def mcmc(data=None,     uncert=None,     func=None,     indparams=None,
         params=None,   pmin=None,       pmax=None,     stepsize=None,
         prior=None,    priorlow=None,   priorup=None,
         nsamples=None, nchains=None,    walk=None,     wlike=None,
         leastsq=None,  chisqscale=None, grtest=None,   burnin=None,
         thinning=None, plots=None,      savefile=None, savemodel=None,
         resume=None,   rms=None,        cfile=None):
  """
  MCMC driver for interactive session.

  Parameters:
  -----------
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
  cfile: String
     Configuration file name.

  Returns:
  --------
  allparams: 2D ndarray
     An array of shape (nfree, nsamples-nchains*burnin) with the MCMC
     posterior distribution of the fitting parameters.
  bestp: 1D ndarray
     Array of the best fitting parameters.

  Notes:
  ------
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
  5.- See the real MCMC code in:
      https://github.com/pcubillos/demc/tree/master/src/mcmc.py

  Examples:
  ---------
  >>> # See examples in: https://github.com/pcubillos/demc/tree/master/examples

  Modification History:
  ---------------------
  2014-05-02  patricio  Initial implementation.
  2014-05-26  patricio  Call now mc3.main with subprocess.
  2014-10-15  patricio  Addded savemodel argument.
  2015-04-25  patricio  Erradicated MPI.  Simplified the whole code.
  """
  # Get function arguments into a dictionary:
  args = locals()
  sys.argv = ['ipython']

  try:
    # Parse configuration file to a dictionary:
    if cfile is not None and not os.path.isfile(cfile):
      mu.error("Configuration file: '{:s}' not found.".format(cfile))
    if cfile:
      config = ConfigParser.SafeConfigParser()
      config.read([cfile])
      defaults = dict(config.items("MCMC"))
    else:
      defaults = {}

    # Get configuration-file values (if any):
    parser = parse()
    parser.set_defaults(**defaults)
    cargs, unknown = parser.parse_known_args()

    # Set undefined argument values:
    for key in args.keys():
      if args[key] is None:
        exec("{:s} = cargs.{:s}".format(key, key))

    # Handle arguments:
    if params is None:
      mu.error("'params' is a required argument.")
    elif isinstance(params[0], str):
      # If params is a filename, unpack:
      if not os.path.isfile(params[0]):
        mu.error("'params' file not found.")
      array = mu.read2array(params[0])
      # Array size:
      ninfo, ndata = np.shape(array)
      if ninfo == 7:                 # The priors
        prior    = array[4]
        priorlow = array[5]
        priorup  = array[6]
      if ninfo >= 4:                 # The stepsize
        stepsize = array[3]
      if ninfo >= 2:                 # The boundaries
        pmin     = array[1]
        pmax     = array[2]
      params = array[0]              # The initial guess

    # Check for pmin and pmax files if not read before:
    if pmin is not None and isinstance(pmin[0], str):
      if not os.path.isfile(pmin[0]):
        mu.error("'pmin' file not found.")
      pmin = mu.read2array(pmin[0])[0]

    if pmax is not None and isinstance(pmax[0], str):
      if not os.path.isfile(pmax[0]):
        mu.error("'pmax' file not found.")
      pmax = mu.read2array(pmax[0])[0]

    # Stepsize:
    if stepsize is not None and isinstance(stepsize[0], str):
      if not os.path.isfile(stepsize[0]):
        mu.error("'stepsize' file not found.")
      stepsize = mu.read2array(stepsize[0])[0]

    # Priors:
    if prior    is not None and isinstance(prior[0], str):
      if not os.path.isfile(prior[0]):
        mu.error("'prior' file not found.")
      prior    = mu.read2array(prior   [0])[0]

    if priorlow is not None and isinstance(priorlow[0], str):
      if not os.path.isfile(priorlow[0]):
        mu.error("'priorlow' file not found.")
      priorlow = mu.read2array(priorlow[0])[0]

    if priorup  is not None and isinstance(priorup[0], str):
      if not os.path.isfile(priorup[0]):
        mu.error("'priorup' file not found.")
      priorup  = mu.read2array(priorup [0])[0]

    # Process the data and uncertainties:
    if data is None:
       mu.error("'data' is a required argument.")
    # If params is a filename, unpack:
    elif isinstance(data[0], str):
      if not os.path.isfile(data[0]):
        mu.error("'data' file not found.")
      array = mu.readbin(data[0])
      data = array[0]
      if len(array) == 2:
        uncert = array[1]

    if uncert is not None and isinstance(uncert[0], str):
      if not os.path.isfile(uncert[0]):
        mu.error("'uncert' file not found.")
      uncert = mu.readbin(uncert[0])[0]

    # Process the independent parameters:
    if indparams != [] and isinstance(indparams[0], str):
      if not os.path.isfile(indparams[0]):
        mu.error("'indparams' file not found.")
      indparams = mu.readbin(indparams[0])

    # Use a copy of uncert to avoid overwrite on it.
    if uncert is not None:
      unc = np.copy(uncert)
    else:
      unc = None

    # Call MCMC:
    allp, bestp = mc.mcmc(data, unc, func, indparams,
                          params, pmin, pmax, stepsize,
                          prior, priorlow, priorup,
                          nsamples, nchains, walk, wlike,
                          leastsq, chisqscale, grtest, burnin,
                          thinning, plots, savefile, savemodel, resume, rms)

    return allp, bestp

  except SystemExit:
    return None


def parse():
  """
  Parse the values from the configuration file.
  """
  # Parse the config file from the command line:
  parser = argparse.ArgumentParser(description=__doc__, #add_help=False,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

  # Configuration-file option:
  parser.add_argument("-c", "--cfile",
                       help="Configuration file.", metavar="FILE")
  # MCMC Options:
  group = parser.add_argument_group("MCMC General Options")
  group.add_argument("-n", "--nsamples",
                     dest="nsamples",
                     help="Number of MCMC samples [default: %(default)s]",
                     type=eval,   action="store", default=int(1e5))
  group.add_argument("-x", "--nchains",
                     dest="nchains",
                     help="Number of chains [default: %(default)s]",
                     type=int,   action="store", default=10)
  group.add_argument("-w", "--walk",
                     dest="walk",
                     help="Random walk algorithm [default: %(default)s]",
                     type=str,   action="store", default="demc",
                     choices=('demc', 'mrw'))
  group.add_argument(      "--wlikelihood",
                     dest="wlike",
                     help="Calculate the likelihood in a wavelet base "
                     "[default: %(default)s]",
                     type=eval,  action="store", default=False)
  group.add_argument(      "--leastsq",
                     dest="leastsq",
                     help="Perform a least-square minimization before the "
                     "MCMC run [default: %(default)s]",
                     type=eval,  action="store", default=False)
  group.add_argument(     "--chisq_scale",
                     dest="chisqscale",
                     help="Scale the data uncertainties such that the reduced "
                     "chi-squared = 1. [default: %(default)s]",
                     type=eval,  action="store", default=False)
  group.add_argument("-g", "--gelman_rubin",
                     dest="grtest",
                     help="Run Gelman-Rubin test [default: %(default)s]",
                     type=eval,  action="store", default=False)
  group.add_argument("-b", "--burnin",
                     help="Number of burn-in iterations (per chain) "
                     "[default: %(default)s]",
                     dest="burnin",
                     type=eval,   action="store", default=0)
  group.add_argument("-t", "--thinning",
                     dest="thinning",
                     help="Chains thinning factor (use every thinning-th "
                     "iteration) for GR test and plots [default: %(default)s]",
                     type=int,     action="store",  default=1)
  group.add_argument(      "--plots",
                     dest="plots",
                     help="If True plot parameter traces, pairwise posteriors, "
                     "and marginal posterior histograms [default: %(default)s]",
                     type=eval,    action="store",  default=False)
  group.add_argument("-o", "--save_file",
                     dest="savefile",
                     help="Output filename to store the parameter posterior "
                     "distributions  [default: %(default)s]",
                     type=str,     action="store",  default="output.npy")
  group.add_argument(      "--savemodel",
                     dest="savemodel",
                     help="Output filename to store the evaluated models  "
                     "[default: %(default)s]",
                     type=str,     action="store",  default=None)
  group.add_argument(      "--resume",
                     dest="resume",
                     help="If True, resume a previous run (load output) "
                     "[default: %(default)s]",
                     type=eval,    action="store",  default=False)
  group.add_argument(      "--rms",
                     dest="rms",
                     help="If True, calculate the RMS of (data-bestmodel) "
                     "[default: %(default)s]",
                     type=eval,    action="store",  default=False)
  group.add_argument("-T", "--tracktime", dest="tractime", action="store_true")
  # Fitting-parameter Options:
  group = parser.add_argument_group("Fitting-function Options")
  group.add_argument("-f", "--func",
                     dest="func",
                     help="List of strings with the function name, module "
                     "name, and path-to-module [required]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-p", "--params",
                     dest="params",
                     help="Filename or list of initial-guess model-fitting "
                     "parameter [required]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-m", "--pmin",
                     dest="pmin",
                     help="Filename or list of parameter lower boundaries "
                     "[default: -inf]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-M", "--pmax",
                     dest="pmax",
                     help="Filename or list of parameter upper boundaries "
                     "[default: +inf]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-s", "--stepsize",
                     dest="stepsize",
                     help="Filename or list with proposal jump scale "
                     "[default: 0.1*params]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-i", "--indparams",
                     dest="indparams",
                     help="Filename or list with independent parameters for "
                     "func [default: None]",
                     type=mu.parray,  action="store", default=[])
  # Data Options:
  group = parser.add_argument_group("Data Options")
  group.add_argument("-d", "--data",
                     dest="data",
                     help="Filename or list of the data being fitted "
                     "[required]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument("-u", "--uncertainties",
                     dest="uncert",
                     help="Filemane or list with the data uncertainties "
                     "[default: ones]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument(     "--prior",
                     dest="prior",
                     help="Filename or list with parameter prior estimates "
                     "[default: %(default)s]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument(     "--priorlow",
                     dest="priorlow",
                     help="Filename or list with prior lower uncertainties "
                     "[default: %(default)s]",
                     type=mu.parray,  action="store", default=None)
  group.add_argument(     "--priorup",
                     dest="priorup",
                     help="Filename or list with prior upper uncertainties "
                     "[default: %(default)s]",
                     type=mu.parray,  action="store", default=None)
  return parser



if __name__ == "__main__":
  warnings.simplefilter("ignore", RuntimeWarning)
  main()


