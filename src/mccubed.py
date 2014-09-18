#! /usr/bin/env python
import sys, os, subprocess, warnings
import argparse, ConfigParser
import timeit
import numpy as np

import mcmc    as mc
import mcutils as mu
start = timeit.default_timer()

def main():
  """
  Multi-Core Markov-Chain Monte Carlo (MC cubed)

  This code calls MCMC to work under an MPI multiprocessor protocol or
  single-thread mode.  When using MPI it will launch one CPU per MCMC chain
  to work in parallel.

  Parameters:
  -----------
  cfile: String
     Filename of a configuration file.

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  2014-05-04  patricio  Added cfile argument for Interpreter support.
  2014-05-26  patricio  Re-engineered the MPI support.
  2014-06-26  patricio  Fixed bug with copy when uncert is None.
  2014-09-14  patricio  Write/read now binary files.
  """

  # Parse the config file from the command line:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file",
                       help="Configuration file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Take configuration file from command-line:
  cfile = args.config_file

  # Incorrect configuration file name:
  if cfile is not None and not os.path.isfile(cfile):
    mu.exit(None, message="Configuration file: '%s' not found."%cfile)
  if cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}

  # Parser for the MCMC arguments:
  parser = argparse.ArgumentParser(parents=[cparser])

  # MCMC Options:
  group = parser.add_argument_group("MCMC General Options")
  group.add_argument("-n", "--numit",
                     dest="numit",
                     help="Number of MCMC samples [default: %(default)s]",
                     type=eval,   action="store", default=100)
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
  group.add_argument(       "--mpi",
                     dest="mpi",
                     help="Run under MPI multiprocessing [default: "
                     "%(default)s]",
                     type=eval,  action="store", default=False)
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

  # Set the defaults from the configuration file:
  parser.set_defaults(**defaults)
  # Set values from command line:
  args2, unknown = parser.parse_known_args(remaining_argv)

  # Unpack configuration-file/command-line arguments:
  numit      = args2.numit
  nchains    = args2.nchains
  walk       = args2.walk
  wlike      = args2.wlike
  leastsq    = args2.leastsq
  chisqscale = args2.chisqscale
  grtest     = args2.grtest
  burnin     = args2.burnin
  thinning   = args2.thinning
  plots      = args2.plots
  savefile   = args2.savefile
  mpi        = args2.mpi
  tracktime  = args2.tractime

  func      = args2.func
  params    = args2.params
  pmin      = args2.pmin
  pmax      = args2.pmax
  stepsize  = args2.stepsize
  indparams = args2.indparams

  data     = args2.data
  uncert   = args2.uncert
  prior    = args2.prior
  priorup  = args2.priorup
  priorlow = args2.priorlow

  nprocs   = nchains

  # Handle arguments:
  if params is None:
    mu.exit(message="'params' is a required argument.")
  elif isinstance(params[0], str):
    # If params is a filename, unpack:
    if not os.path.isfile(params[0]):
      mu.exit(message="'params' file not found.")
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
      mu.exit(message="'pmin' file not found.")
    pmin = mu.read2array(pmin[0])[0]

  if pmax is not None and isinstance(pmax[0], str):
    if not os.path.isfile(pmax[0]):
      mu.exit(message="'pmax' file not found.")
    pmax = mu.read2array(pmax[0])[0]

  # Stepsize:
  if stepsize is not None and isinstance(stepsize[0], str):
    if not os.path.isfile(stepsize[0]):
      mu.exit(message="'stepsize' file not found.")
    stepsize = mu.read2array(stepsize[0])[0]

  # Priors:
  if prior    is not None and isinstance(prior[0], str):
    if not os.path.isfile(prior[0]):
      mu.exit(message="'prior' file not found.")
    prior    = mu.read2array(prior   [0])[0]

  if priorlow is not None and isinstance(priorlow[0], str):
    if not os.path.isfile(priorlow[0]):
      mu.exit(message="'priorlow' file not found.")
    priorlow = mu.read2array(priorlow[0])[0]

  if priorup  is not None and isinstance(priorup[0], str):
    if not os.path.isfile(priorup[0]):
      mu.exit(message="'priorup' file not found.")
    priorup  = mu.read2array(priorup [0])[0]

  # Process the data and uncertainties:
  if data is None:
     mu.exit(comm, True, "'data' is a required argument.")
  # If params is a filename, unpack:
  elif isinstance(data[0], str):
    if not os.path.isfile(data[0]):
      mu.exit(message="'data' file not found.")
    array = mu.readbin(data[0])
    data = array[0]
    if len(array) == 2:
      uncert = array[1]

  if uncert is not None and isinstance(uncert[0], str):
    if not os.path.isfile(uncert[0]):
      mu.exit(message="'uncert' file not found.")
    uncert = mu.readbin(uncert[0])[0]

  # Process the independent parameters:
  if indparams != [] and isinstance(indparams[0], str):
    if not os.path.isfile(indparams[0]):
      mu.exit(message="'indparams' file not found.")
    indparams = mu.readbin(indparams[0])

  if tracktime:
    start_mpi = timeit.default_timer()

  if mpi:
    # Checks for mpi4py:
    try:
      from mpi4py import MPI
    except:
      mu.exit(message="Attempted to use MPI, but mpi4py is not installed.")

    # Get source dir:
    mcfile = mc.__file__
    iright = mcfile.rfind('/')
    if iright == -1:
      sdir = "."
    else:
      sdir = mcfile[:iright]

    # Call wrapper of model function:
    args = [sdir+"/func.py", "-c"+cfile] + remaining_argv
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=nprocs)
  else:
    comm = None

  # Use a copy of uncert to avoid overwrite on it.
  if uncert is not None:
    unc = np.copy(uncert)
  else:
    unc = None

  if tracktime:
    start_loop = timeit.default_timer()
  # Run the MCMC:
  allp, bp = mc.mcmc(data, unc, func, indparams,
                     params, pmin, pmax, stepsize,
                     prior, priorlow, priorup,
                     numit, nchains, walk, wlike,
                     leastsq, chisqscale, grtest, burnin,
                     thinning, plots, savefile, comm)

  if tracktime:
    stop = timeit.default_timer()

  # Close communications and disconnect:
  if mpi:
    mu.comm_disconnect(comm)

  #if bench == True:
  if tracktime:
    print("Total execution time:   %10.6f s"%(stop - start))
    print("Time to initialize MPI: %10.6f s"%(start_loop - start_mpi))


def mcmc(data=None,     uncert=None,     func=None,     indparams=None,
         params=None,   pmin=None,       pmax=None,     stepsize=None,
         prior=None,    priorlow=None,   priorup=None,
         numit=None,    nchains=None,    walk=None,     wlike=None,
         leastsq=None,  chisqscale=None, grtest=None,   burnin=None,
         thinning=None, plots=None,      savefile=None, mpi=None, cfile=False):
  """
  MCMC wrapper for interactive session.

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
  numit: Scalar
     Total number of iterations.
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
  mpi: Boolean
     If True run under MPI multiprocessing protocol (not available in
     interactive mode).
  cfile: String
     Configuration file name.

  Returns:
  --------
  allparams: 2D ndarray
     An array of shape (nfree, numit-nchains*burnin) with the MCMC
     posterior distribution of the fitting parameters.
  bestp: 1D ndarray
     Array of the best fitting parameters.

  Notes:
  ------
  1.- If a value is 0, keep the parameter fixed.
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
  """
  sys.argv = ['ipython']

  try:
    # Store arguments in a dict:
    piargs = {}
    piargs.update({'data':     data})
    piargs.update({'uncert':   uncert})
    piargs.update({'func':     func})
    piargs.update({'indparams':indparams})
    piargs.update({'params':   params})
    piargs.update({'pmin':     pmin})
    piargs.update({'pmax':     pmax})
    piargs.update({'stepsize': stepsize})
    piargs.update({'prior':    prior})
    piargs.update({'priorlow': priorlow})
    piargs.update({'priorup':  priorup})
    piargs.update({'numit':    numit})
    piargs.update({'nchains':  nchains})
    piargs.update({'walk':     walk})
    piargs.update({'wlike':    wlike})
    piargs.update({'leastsq':  leastsq})
    piargs.update({'chisqscale': chisqscale})
    piargs.update({'grtest':   grtest})
    piargs.update({'burnin':   burnin})
    piargs.update({'thinning': thinning})
    piargs.update({'plots':    plots})
    piargs.update({'savefile': savefile})
    piargs.update({'mpi':      mpi})

    # Remove None values:
    for key in piargs.keys():
      if piargs[key] is None:
        piargs.pop(key)

    # Temporary files:
    tmpfiles = []
    # Open ConfigParser:
    config = ConfigParser.SafeConfigParser()
    if not cfile:
      config.add_section('MCMC')  # Start new config file
    else:
      config.read(cfile)          # Read from existing config file

    # Store arguments in configuration file:
    for key in piargs.keys():
      value = piargs[key]
      # Func:
      if   key == 'func':
        if callable(func):
          funcfile = func.__globals__['__file__']
          funcpath = funcfile[:funcfile.rfind('/')]
          config.set('MCMC', key, "%s %s %s"%(func.__name__,
                                              func.__module__, funcpath))
        else:
          config.set('MCMC', key, " ".join(func))
      # Arrays:
      elif key in ['data', 'uncert', 'indparams', 'params', 'pmin', 'pmax',
                   'stepsize', 'prior', 'priorlow', 'priorup']:
        if isinstance(value, str):
          config.set('MCMC', key, value)
        else:
          arrfile = "temp_mc3_mpi_%s.dat"%key   # Set file name to store array
          if key in ['data', 'uncert']:
            mu.writebin([value], arrfile) # Write array into file
          elif key in ['indparams']:
            mu.writebin(value, arrfile)         # Write array into file
          else:
            mu.writedata(value, arrfile)        # Write array into file
          config.set('MCMC', key, arrfile)      # Set filename in config
          tmpfiles.append(arrfile)
      # Everything else:
      else:
        config.set('MCMC', key, str(value))

    # Get/set the output file:
    if piargs.has_key('savefile'):
      savefile = piargs['savefile']
    elif config.has_option('MCMC', 'savefile'):
      savefile = config.get('MCMC', 'savefile')
    else:
      savefile = 'temp_mc3_mpi_savefile.npy'
      config.set('MCMC', 'savefile', savefile)
      tmpfiles.append(savefile)


    # Save the configuration file:
    cfile = 'temp_mc3_mpi_configfile.cfg'
    tmpfiles.append(cfile)
    with open(cfile, 'wb') as configfile:
      config.write(configfile)
    piargs.update({'cfile':cfile})

    # Call main:
    subprocess.call(["mpirun %s -c %s"%(os.path.realpath(__file__).rstrip("c"),
                     cfile)], shell=True)

    # Read output:
    allp = np.load(savefile)
    bestp = allp.T[-1]

    # Remove temporary files:
    for file in tmpfiles:
      os.remove(file)

    return allp, bestp

  except SystemExit:
    pass


if __name__ == "__main__":
  warnings.simplefilter("ignore", RuntimeWarning)
  main()
