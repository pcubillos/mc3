#! /usr/bin/env python
import os, sys, warnings
import argparse, ConfigParser
import numpy as np
import matplotlib.pyplot as plt

import gelman_rubin as gr
import mcutils as mu
import mcplots as mp

def mcmc(data, uncert=None, func=None, indparams=[],
         params=None, pmin=None, pmax=None, stepsize=None,
         prior=None, priorup=None, priorlow=None,
         numit=10, nchains=10, walk='demc',
         grtest=True, burnin=0, thinning=1,
         plots=False, savefile=None, mpi=False):
  """
  This beautiful piece of code runs a Markov-chain Monte Carlo algoritm.

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
  priorup: 1D ndarray
     Upper prior uncertainty values (See Note 2).
  priorlow: 1D ndarray
     Lower prior uncertainty values (See Note 2).
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

  Returns:
  --------
  allparams: 2D ndarray
     An array of shape (nfree, numit-nchains*burnin) with the MCMC
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

  Examples:
  ---------
  >>> # See examples in: https://github.com/pcubillos/demc/tree/master/examples

  Modification History:
  ---------------------
    2008-05-02  Written by:  Kevin Stevenson, UCF
                             kevin218@knights.ucf.edu
    2008-06-21  kevin     Finished updating
    2009-11-01  kevin     Updated for multi events:
    2010-06-09  kevin     Updated for ipspline, nnint & bilinint
    2011-07-06  kevin     Updated for Gelman-Rubin statistic
    2011-07-22  kevin     Added principal component analysis
    2011-10-11  kevin     Added priors
    2012-09-03  patricio  Added Differential Evolution MC. Documented.
                          pcubillos@fulbrightmail.org, UCF
    2013-01-31  patricio  Modified for general purposes.
    2013-02-21  patricio  Added support distribution for DEMC.
    2014-03-31  patricio  Modified to be completely agnostic of the
                          fitting function, updated documentation.
    2014-04-17  patricio  Revamped use of 'func': no longer requires a
                          wrapper.  Alternatively, can take a string list with
                          the function, module, and path names.
    2014-04-19  patricio  Added savefile, thinning, plots, and mpi arguments.
  """

  # Import the model function:
  if type(func) in [list, tuple, np.ndarray]:
    if len(func) == 3:
      sys.path.append(func[2])
    exec('from %s import %s as func'%(func[1], func[0]))
  elif not callable(func):
    print("'func' must be a callable or an iterable (list, tuple, or ndarray) "
          "\n of strings with the model function, file, and path names.")
    sys.exit(0)

  ndata     = len(data)
  nparams   = len(stepsize)              # Number of model params
  nfree     = np.sum(stepsize > 0)       # Number of free parameters
  chainlen  = np.ceil(numit/nchains)     # Number of iterations per chain
  ifree     = np.where(stepsize > 0)[0]  # Free   parameter indices
  ishare    = np.where(stepsize < 0)[0]  # Shared parameter indices

  # Set default uncertainties:
  if uncert is None:
    uncert = np.ones(ndata)
  # Set default boundaries:
  if pmin is None:
    pmin = np.zeros(ndata) - np.inf
  if pmax is None:
    pmax = np.zeros(ndata) + np.inf
  # Set default stepsize:
  if stepsize is None:
    stepsize = 0.1 * params
  # Set prior parameter indices:
  if (prior or priorup or priorlow) is None:
    iprior = np.array([])  # Empty array
  else:
    iprior  = np.where(priorup  > 0)[0]
  # Intermediate steps to run GR test and print progress report
  intsteps  = int(chainlen / 10)
  numaccept = np.zeros(nchains)          # Number of accepted proposal jumps
  outbounds = np.zeros((nchains, nfree)) # Number of out of bounds proposals 
  allparams = np.zeros((nchains, nfree, chainlen)) # Parameter's record

  if mpi:
    # Send sizes info to other processes:
    array1 = np.asarray([nparams, ndata, chainlen], np.int)
    mu.comm_gather(comm, array1, MPI.INT)

  # DEMC parameters:
  gamma  = 2.4 / np.sqrt(2*nfree)
  gamma2 = 0.01  # Jump scale factor of support distribution

  # Make params 2D shaped (nchains, nparams):
  if np.ndim(params) == 1:
    params = np.repeat(np.atleast_2d(params), nchains, 0)
    # Start chains with an initial jump:
    for p in ifree:
      # For each free param, use a normal distribution: 
      params[1:, p] = np.random.normal(params[0, p], stepsize[p], nchains-1)
      # Stay within pmin and pmax boundaries:
      params[np.where(params[:, p] < pmin[p]), p] = pmin[p]
      params[np.where(params[:, p] > pmax[p]), p] = pmax[p]
  
  # Update shared parameters:
  for s in ishare:
    params[:, s] = params[:, -int(stepsize[s])-1]

  # Calculate chi-squared for model type using current params:
  models = np.zeros((nchains, ndata))
  if mpi:
    # Gather (send) parameters to hub:
    mu.comm_gather(comm, params.flatten(), MPI.DOUBLE)
    # Scatter (receive) evaluated models:
    mpimodels = np.zeros(nchains*ndata, np.double)
    mu.comm_scatter(comm, mpimodels)
    # Store them in models variable:
    models = np.reshape(mpimodels, (nchains, ndata))
  else:
    for c in np.arange(nchains):
      fargs = [params[c]] + indparams  # List of function's arguments
      models[c] = func(*fargs)

  # Calculate chi square for each chain:
  currchisq = np.zeros(nchains)
  for c in np.arange(nchains):
    currchisq[c] = np.sum( ((models[c]-data)/uncert)**2.0 )
    # Apply prior, if exists:
    if len(iprior) > 0:
      pdiff  = params[c] - prior   # prior difference
      psigma = np.zeros(nparams)   # prior standard deviation
      # Determine psigma based on which side of the prior is the param:
      psigma[np.where(pdiff >  0)] = priorup [np.where(pdiff >  0)]
      psigma[np.where(pdiff <= 0)] = priorlow[np.where(pdiff <= 0)]
      currchisq[c] += np.sum((pdiff/psigma)[iprior]**2.0)

  # Get lowest chi-square and best fitting parameters:
  bestchisq = np.amin(currchisq)
  bestp     = params[np.argmin(currchisq)]

  # Set up the random walks:
  if   walk == "mrw":
    # Generate proposal jumps from Normal Distribution for MRW:
    mstep   = np.random.normal(0, stepsize[ifree], (chainlen, nchains, nfree))
  elif walk == "demc":
    # Support random distribution:
    support = np.random.normal(0, stepsize[ifree], (chainlen, nchains, nfree))
    # Generate indices for the chains such r[c] != c:
    r1 = np.random.randint(0, nchains-1, (nchains, chainlen))
    r2 = np.random.randint(0, nchains-1, (nchains, chainlen))
    for c in np.arange(nchains):
      r1[c][np.where(r1[c]==c)] = nchains-1
      r2[c][np.where(r2[c]==c)] = nchains-1

  # Uniform random distribution for the Metropolis acceptance rule:
  unif = np.random.uniform(0, 1, (chainlen, nchains))

  # Proposed iteration parameters and chi-square (per chain):
  nextp     = np.copy(params)    # Proposed parameters
  nextchisq = np.zeros(nchains)  # Chi square of nextp 

  # Start loop:
  for i in np.arange(chainlen):
    # Proposal jump:
    if   walk == "mrw":
      jump = mstep[i]
    elif walk == "demc":
      jump = (gamma  * (params[r1[:,i]]-params[r2[:,i]])[:,ifree] +
              gamma2 * support[i]                                 )
    # Propose next point:
    nextp[:,ifree] = params[:,ifree] + jump

    # Check it's within boundaries: 
    outbounds += ((nextp < pmin) | (nextp > pmax))[:,ifree]
    for p in ifree:
      nextp[np.where(nextp[:, p] < pmin[p]), p] = pmin[p]
      nextp[np.where(nextp[:, p] > pmax[p]), p] = pmax[p]

    # Update shared parameters:
    for s in ishare:
      nextp[:, s] = nextp[:, -int(stepsize[s])-1]

    # Evaluate the models for the proposed parameters:
    if mpi:
      mu.comm_gather(comm, nextp.flatten(), MPI.DOUBLE)
      mu.comm_scatter(comm, mpimodels)
      models = np.reshape(mpimodels, (nchains, ndata))
    else:
      for c in np.arange(nchains):
        fargs = [nextp[c]] + indparams  # List of function's arguments
        models[c] = func(*fargs)

    # Calculate chisq:
    for c in np.arange(nchains):
      nextchisq[c] = np.sum(((models[c]-data)/uncert)**2.0) 
      # Apply prior:
      if len(iprior) > 0:
        pdiff  = nextp[c] - prior    # prior difference
        psigma = np.zeros(nparams)   # prior standard deviation
        # Determine psigma based on which side of the prior is nextp:
        psigma[np.where(pdiff >  0)] = priorup [np.where(pdiff >  0)]
        psigma[np.where(pdiff <= 0)] = priorlow[np.where(pdiff <= 0)]
        nextchisq[c] += np.sum((pdiff/psigma)[iprior]**2.0)

    # Evaluate which steps are accepted and update values:
    accept = np.exp(0.5 * (currchisq - nextchisq))
    accepted = accept >= unif[i]
    numaccept += accepted
    # Update params and chi square:
    params   [accepted] = nextp    [accepted]
    currchisq[accepted] = nextchisq[accepted]

    # Check lowest chi-square:
    if np.amin(currchisq) < bestchisq:
      bestp = params[np.argmin(currchisq)]
      bestchisq = np.amin(currchisq)
    # Store current iteration values:
    allparams[:,:,i] = params[:, ifree]
  
    # Print intermediate info:
    if ((i+1) % intsteps == 0) and (i > 0):
      mu.progressbar((i+1.0)/chainlen)
      print("Out-of-bound Trials: ")
      print(np.sum(outbounds, axis=0))
      print("Best Parameters:\n%s   (chisq=%.4f)"%(str(bestp), bestchisq))

      # Gelman-Rubin statistic:
      if grtest and i > burnin:
        psrf = gr.convergetest(allparams[:, ifree, burnin:i+1:thinning])
        print("Gelman-Rubin statistic for free parameters:\n" + str(psrf))
        if np.all(psrf < 1.01):
          print("All parameters have converged to within 1% of unity.")

  print("Finito.")
  # Stack together the chains:
  allstack = allparams[0, :, burnin:]
  for c in np.arange(1, nchains):
    allstack = np.hstack((allstack, allparams[c, :, burnin:]))

  if plots:
    # Extract filename from savefile:
    if savefile is not None:
      if savefile.rfind(".") == -1:
        fname = savefile[savefile.rfind("/")+1:]
      else:
        fname = savefile[savefile.rfind("/")+1:savefile.rfind(".")]
    else:
      fname = "MCMC"
    # Trace plot:
    mp.trace(allstack,     thinning=thinning, savefile=fname+"_trace.pdf")
    # Pairwise posteriors:
    mp.pairwise(allstack,  thinning=thinning, savefile=fname+"_pairwise.pdf")
    # Histograms:
    mp.histogram(allstack, thinning=thinning, savefile=fname+"_posterior.pdf")

  if savefile is not None:
    output = open(savefile, 'w')
    np.save(output, allstack)
    output.close()

  return allstack, bestp


def main(comm):
  """
  Take arguments from the command line and run MCMC when called from the prompt

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """

  # Initialise parser to process a configuration file:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file", type=str,
                       help="Configuration file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file (if exists):
  cfile = args.config_file # The configuration file
  if cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}

  # Now, parser for the MCMC arguments:
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
  group.add_argument("-o", "--output_file",
                     dest="output",
                     help="Output filename to store the parameter posterior "
                     "distributions  [default: %(default)s]",
                     type=str,     action="store",  default="output.npy")
  group.add_argument(       "--mpi",
                     dest="mpi",
                     help="Run under MPI multiprocessing [default: "
                     "%(default)s]",
                     type=eval,  action="store", default=False)
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

  # Unpack command-line-arguments:
  numit    = args2.numit
  nchains  = args2.nchains
  walk     = args2.walk
  grtest   = args2.grtest
  burnin   = args2.burnin
  thinning = args2.thinning
  plots    = args2.plots
  output   = args2.output
  mpi      = args2.mpi

  func     = args2.func
  params   = args2.params
  pmin     = args2.pmin
  pmax     = args2.pmax
  stepsize = args2.stepsize
  indparams = args2.indparams

  data     = args2.data
  uncert   = args2.uncert
  prior    = args2.prior
  priorup  = args2.priorup
  priorlow = args2.priorlow

  # Checks for mpi4py:
  if mpi:
    if comm is None:
      mu.exit(message="Attempted to use MPI, but mpi4py is not installed.")
    try:
      commname = comm.Get_name()
    except:
      mu.exit(None, message="Invalid communicator.  Did you run mcmc.py? "
                            "For MPI run mpmc.py instead.")
  if not mpi:
    comm = None

  # Handle arguments:
  if params is None:
    mu.exit(comm, True, "'params' is a required argument.")
  elif isinstance(params[0], str):
    # If params is a filename, unpack:  
    if not os.path.isfile(params[0]):
      mu.exit(comm, True, "'params' file not found.")
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
      mu.exit(comm, True, "'pmin' file not found.")
    pmin = mu.read2array(pmin[0])[0]

  if pmax is not None and isinstance(pmax[0], str):
    if not os.path.isfile(pmax[0]):
      mu.exit(comm, True, "'pmax' file not found.")
    pmax = mu.read2array(pmax[0])[0]

  # Stepsize:
  if stepsize is not None and isinstance(stepsize[0], str):
    if not os.path.isfile(stepsize[0]):
      mu.exit(comm, True, "'stepsize' file not found.")
    stepsize = mu.read2array(stepsize[0])[0]

  # Priors:
  if prior    is not None and isinstance(prior[0], str):
    if not os.path.isfile(prior[0]):
      mu.exit(comm, True, "'prior' file not found.")
    prior    = mu.read2array(prior   [0])[0]

  if priorlow is not None and isinstance(priorlow[0], str):
    if not os.path.isfile(priorlow[0]):
      mu.exit(comm, True, "'priorlow' file not found.")
    priorlow = mu.read2array(priorlow[0])[0]

  if priorup  is not None and isinstance(priorup[0], str):
    if not os.path.isfile(priorup[0]):
      mu.exit(comm, True, "'priorup' file not found.")
    priorup  = mu.read2array(priorup [0])[0]

  # Process the data and uncertainties:
  if data is None:
     mu.exit(comm, True, "'data' is a required argument.")
  # If params is a filename, unpack:  
  elif isinstance(data[0], str):
    if not os.path.isfile(data[0]):
      mu.exit(comm, True, "'data' file not found.")
    array = mu.read2array(data[0])
    # Array size:
    ninfo, ndata = np.shape(array)
    data = array[0]
    if ninfo == 2:
      uncert = array[1]

  if uncert is not None and isinstance(uncert[0], str):
    if not os.path.isfile(uncert[0]):
      mu.exit(comm, True, "'uncert' file not found.")
    uncert = mu.read2array(uncert[0])[0]

  # Process the independent parameters:
  if indparams != [] and isinstance(indparams[0], str):
    if not os.path.isfile(indparams[0]):
      mu.exit(comm, True, "'indparams' file not found.")
    indparams = mu.read2array(indparams[0], square=False)

  # Send OK:
  if mpi:
    mu.comm_gather(comm, np.array([0]), MPI.INT)

  # Run the MCMC:
  allp, bp = mcmc(data, uncert, func, indparams,
                  params, pmin, pmax, stepsize,
                  prior, priorup, priorlow,
                  numit, nchains, walk, grtest, burnin,
                  thinning, plots, output, mpi)

  print("The best-fit parameters are:  " + str(bp)               +
      "\nThe mean parameters are:      " + str(np.mean(allp, 1)) +
      "\nWith uncertainties:           " + str(np.std(allp,  1)))

  # Successful exit
  mu.exit(comm)


if __name__ == "__main__":
  warnings.simplefilter("ignore", RuntimeWarning)
  try:
    from mpi4py import MPI
    comm = MPI.Comm.Get_parent()
  except:
    comm = None
  main(comm)
