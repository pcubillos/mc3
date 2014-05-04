#! /usr/bin/env python
import sys, os, subprocess
import argparse, ConfigParser
import timeit
import numpy as np
import mcmc as mc
from mpi4py import MPI

import mcutils as mu
start = timeit.default_timer()

def main(cfile=None):
  """
  Multi-Processor Markov-Chain Monte Carlo (MPMC)

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
  """
  # Parse arguments:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file",
                       help="Specify config file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Take configuration file from command-line if not given as an argument:
  if cfile is None:
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
  parser = argparse.ArgumentParser(parents=[cparser], add_help=False)
  parser.add_argument("-x", "--nchains",   dest="nchains", type=int,
                                           action="store", default=10)
  parser.add_argument(       "--mpi",      dest="mpi",     type=eval,
                    help="Run under MPI multiprocessing [default %(default)s]",
                     action="store", default=False)
  parser.add_argument("-T", "--tracktime", dest="tractime", action="store_true")
  parser.add_argument("-h", "--help",      dest="help", action="store_true")

  parser.set_defaults(**defaults)
  args2, unknown = parser.parse_known_args(remaining_argv)

  # The number of processors is the number of chains:
  nprocs    = args2.nchains
  mpi       = args2.mpi
  # Hidden feature to track the execution of the time per loop for MPI:
  tracktime = args2.tractime

  # Get source dir:
  mcfile = mc.__file__
  iright = mcfile.rfind('/')
  if iright == -1:
    sdir = "."
  else:
    sdir = mcfile[:iright]

  # If asked for help:
  if args2.help:
    subprocess.call([sdir + "/mcmc.py --help"], shell=True)
    sys.exit(0)

  if not mpi:
    subprocess.call([sdir+"/mcmc.py " + " ".join(sys.argv[1:])], shell=True)
  else:
    if tracktime:
      start_mpi = timeit.default_timer()
    # Call MCMC:
    args = [sdir+"/mcmc.py", "-c"+cfile] + remaining_argv
    comm1 = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=1)

    # Get OK flag from MCMC:
    abort = np.array([0])
    mu.comm_gather(comm1, abort)
    if abort[0]:
      mu.exit(comm1)

    # Call wrapper of model function:
    args = [sdir+"/func.py", "-c"+cfile] + remaining_argv
    comm2 = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=nprocs)

    # MPI get sizes from MCMC:
    array1 = np.zeros(3, np.int)
    mu.comm_gather(comm1, array1)
    npars, ndata, niter = array1
    # MPI Broadcast to workers:
    mu.comm_bcast(comm2, np.asarray([npars, niter], np.int), MPI.INT)

    # get npars, ndata from MCMC:
    mpipars   = np.zeros(npars*nprocs, np.double)
    mpimodels = np.zeros(ndata*nprocs, np.double)

    if tracktime:
      start_loop = timeit.default_timer()
      loop_timer = []
      loop_timer2 = []
    while niter >= 0:
      if tracktime:
        loop_timer.append(timeit.default_timer())
      # Gather (receive) parameters from MCMC:
      mu.comm_gather(comm1, mpipars)

      # Scatter (send) parameters to funcwrapper:
      mu.comm_scatter(comm2, mpipars, MPI.DOUBLE)
      # Gather (receive) models:
      mu.comm_gather(comm2, mpimodels)

      # Scatter (send) results to MCMC:
      mu.comm_scatter(comm1, mpimodels, MPI.DOUBLE)
      niter -= 1
      if tracktime:
        loop_timer2.append(timeit.default_timer() - loop_timer[-1])

    if tracktime:
      stop = timeit.default_timer()

    # Close communications and disconnect:
    if mpi:
      mu.comm_disconnect(comm1)
      mu.comm_disconnect(comm2)

    #if bench == True:
    if tracktime:
      print("Total execution time:   %10.6f s"%(stop - start))
      print("Time to initialize MPI: %10.6f s"%(start_loop - start_mpi))
      print("Time to run first loop: %10.6f s"%(loop_timer[1] - loop_timer[0]))
      print("Time to run last loop:  %10.6f s"%(loop_timer[-1]- loop_timer[-2]))
      print("Time to run avg loop:   %10.6f s"%(np.mean(loop_timer2)))


def mcmc(data=None,   uncert=None,   func=None,  indparams=None,
         params=None, pmin=None,     pmax=None,  stepsize=None,
         prior=None,  priorlow=None, priorup=None,
         numit=None,  nchains=None,  walk=None,
         grtest=None, burnin=None,   thinning=None,
         plots=None,  savefile=None, mpi=None,   cfile=False):
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

    if mpi is None or not mpi:
      # Always take value of cfile (even if not set by user):
      piargs.update({'cfile':cfile})

      # Store these in a list if they are a path-to-file string:
      for key in piargs.keys():
        value = piargs[key]
        if key in ['data', 'uncert', 'indparams', 'params', 'pmin', 'pmax',
                   'stepsize', 'prior', 'priorlow', 'priorup']:
          if isinstance(value, str):
            piargs.update({key:[value]})

      # All parameters, best parameters:
      allp, bestp = mc.main(None, piargs)

    # mpi is True:
    else:
      # Temporary files:
      tmpfiles = []
      # Open ConfigParser:
      config = ConfigParser.SafeConfigParser()
      if not cfile:   # Start new config file
        config.add_section('MCMC')
      else:               # Read from existing config file
        config.read(cfile)

      # Store values in configuration file:
      for key in piargs.keys():
        value = piargs[key]
        if   key == 'func':
          if callable(func):
            funcfile = func.__globals__['__file__']
            funcpath = funcfile[:funcfile.rfind('/')]
            config.set('MCMC', key, "%s %s %s"%(func.__name__,
                                                func.__module__, funcpath))
          else:
            config.set('MCMC', key, " ".join(func))
        elif key in ['data', 'uncert', 'indparams', 'params', 'pmin', 'pmax',
                     'stepsize', 'prior', 'priorlow', 'priorup']:
          if not isinstance(value, str):
            arrfile = "temp_mc3_mpi_%s.dat"%key # Set file name to store array
            if key == 'indparams':
              mu.writedata(value, arrfile, True) # Write array into file
            else:
              mu.writedata(value, arrfile)     # Write array into file
            config.set('MCMC', key, arrfile)    # Set filename in config
            tmpfiles.append(arrfile)
          else:
            config.set('MCMC', key, value)
        else:
          config.set('MCMC', key, str(value))

      # Set a output file if there was not one:
      if not config.has_option('MCMC', 'savefile'):
        savefile = 'temp_mc3_mpi_savefile.npy'
        config.set('MCMC', 'savefile', savefile)
        tmpfiles.append(savefile)
      else:
        savefile = config.get('MCMC', 'savefile')

      # Save the configuration file:
      cfile = 'temp_mc3_mpi_configfile.cfg'
      tmpfiles.append(cfile)
      with open(cfile, 'wb') as configfile:
        config.write(configfile)

      # Call main:
      main(cfile)

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
  main()
