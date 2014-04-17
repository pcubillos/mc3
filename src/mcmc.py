#! /usr/bin/env python
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import gelman_rubin as gr

def mcmc(data, uncert, func, indparams,
         params, pmin, pmax, stepsize,
         prior=None, priorup=None, priorlow=None,
         numit=10, nchains=10, walk='demc', grtest=True, burnin=0):
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
        model = func(params, indparams)
     Or an iterable (list, tuple, or ndarray) of 3 strings:
        (funcname, modulename, path)
     that specify the function name, function module, and module path.
     If the module is already in the python-path scope, path can be omitted.
  indparams: tuple
     Additional arguments given to func.
  params: 1D or 2D ndarray
     Set of initial fitting parameters for func.  If 2D, of shape
     (nparams, nchains), it is assumed that it is one set for each chain.
  pmin: 1D ndarray
     Lower boundaries of the posteriors.
  pmax: 1D ndarray
     Upper boundaries of the posteriors.
  stepsize: 1D ndarray
     Proposal jump scale.  If a values is 0, keep the parameter fixed.
     Negativevalues indicate a shared parameter (See Note 1).
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
    2014-04-17  patricio  Revamped use of 'func': no longer requires a wrapper,
                          Alternatively, can take a string list with the
                          function, module, and path names.
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
  # Set prior parameter indices:
  if (prior or priorup or priorlow) is None:
    iprior = np.array([])  # Empty array
  else:
    iprior  = np.where(priorup  > 0)[0]
  intsteps  = int(chainlen / 10)
  numaccept = np.zeros(nchains)          # Number of accepted proposal jumps
  outbounds = np.zeros((nchains, nfree)) # Number of out of bounds proposals 
  allparams = np.zeros((nchains, nfree, chainlen)) # Parameters record

  #print("ifree:  " + str(ifree))
  #print("ishare: " + str(ishare))
  #print("iprior: " + str(iprior))
  #print("nparams: %d,  nfree: %d,  nchains: %d,  chainlen: %d"%(
  #       nparams, nfree, nchains, chainlen))

  # DEMC parameters:
  gamma  = 2.4 / np.sqrt(2*nfree)
  gamma2 = 0.01  # Jump scale factor or support distribution

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

  #print("params: \n" + str(params))

  # Send initialization info to workers:
  #init_info = np.array([fs, df, nfreq, smin, pmax[1], chainlen, nfree])
  #icomm.Bcast(init_info, root=MPI.ROOT)

  # Calculate chi-squared for model type using current params:
  # The function 'func' should encapsulate all the maja-mama to calculate
  # the models for each chain, whether it's a MPI call or a 'normal'
  # function call. It should return a list of models, one for each chain. 
  models = np.zeros((nchains, ndata))
  for c in np.arange(nchains):
    models[c] = func(params[c], indparams)


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
  print("Chisq:" + str(currchisq))
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

  # Proposed iteration chi-square per chain:
  nextp     = np.copy(params)    # Proposed parameters
  nextchisq = np.zeros(nchains)  # Chi square of nextp 
  # Start loop:
  for i in np.arange(chainlen):
    # Propose next point:
    if   walk == "mrw":
      jump = mstep[i]
    elif walk == "demc":
      jump = ( gamma  * (params[r1[:,i]]-params[r2[:,i]])[:,ifree] +
               gamma2 * support[i]                                 )

    # Proposal jump:
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
    for c in np.arange(nchains):
      models[c] = func(nextp[c], indparams)

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

    allparams[:,:,i] = params[:, ifree]
  
    # Print intermediate info:
    if ((i+1) % intsteps == 0) and (i > 0):
      print("\n" + time.ctime())
      print("P: %04d  best chisq = %12.6f  (%6.3f, %5.3f, %7.2f)"%(
            i, bestchisq, bestp[0], bestp[1], bestp[2]))
      print("Number of times parameter tries to step out of bounds: ")
      print(np.sum(outbounds, axis=0))
      print("Current Best Parameters:\n" + str(bestp))

      # Gelman-Rubin statistic:
      if grtest and i > burnin:
        psrf = gr.convergetest(allparams[:, ifree, burnin:i+1])
        print("Gelman-Rubin statistic for free parameters:\n" + str(psrf))
        if np.all(psrf < 1.01):
          print("All parameters have converged to within 1% of unity.\n")

  print("P: Finito")
  # Stack together the chains:
  allstack = allparams[0, :, burnin:]
  for c in np.arange(1, nchains):
    allstack = np.hstack((allstack, allparams[c, :, burnin:]))
  return allstack, bestp
