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
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
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

import sys, os, ConfigParser
import numpy   as np

import mcutils  as mu
import modelfit as mf

def prayer(configfile, nprays=0, savefile=None):
  """
  Implement prayer bead method to estimate parameter uncertainties.

  Parameters:
  -----------
  params: 1D-ndarray 
    Comment me, and all my friends.
  inonprior: 1D-ndarray
  stepsize: 1D-ndarray
  fit: a fits instance
  ncores: integer

  Notes:
  ------
  Believing in a prayer bead is a mere act of faith, we are scientists
  for god's sake!

  Modification History:
  ---------------------
  2012-10-29  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  2013-09-03  patricio  Added documentation.  
  2014-05-19  patricio  Modified to work with MC3.
  """

  config = ConfigParser.SafeConfigParser()
  config.read([configfile])
  cfgsec = "MCMC" 

  data = mu.parray(config.get(cfgsec, 'data'))
  if isinstance(data[0], str):
    array = mu.readbin(data[0])
    data = array[0]
    if len(array) == 2:
      uncert = array[1]
    else:
      uncert = mu.parray(config.get(cfgsec, 'uncert'))

  params    = mu.parray(config.get(cfgsec, 'params'))
  if isinstance(params[0], str):
    array = mu.read2array(params[0])
    ninfo, nparams = np.shape(array)
    if ninfo == 7:                 # The priors
      prior    = array[4]
      priorlow = array[5]
      priorup  = array[6]
    else:
      try:
        prior     = mu.parray(config.get(cfgsec, 'prior'))
        priorlow  = mu.parray(config.get(cfgsec, 'priorlow'))
        priorup   = mu.parray(config.get(cfgsec, 'priorup'))
      except:
        prior   = np.zeros(nparams)  # Empty arrays
        priorup = priorlow = np.array([])
        iprior  = np.array([], int)

    if ninfo >= 4:                 # The stepsize
      stepsize = array[3]
    else:
      stepsize  = mu.parray(config.get(cfgsec, 'stepsize'))

    if ninfo >= 2:                 # The boundaries
      pmin     = array[1]
      pmax     = array[2]
    else:
      pmin      = mu.parray(config.get(cfgsec, 'pmin'))
      pmax      = mu.parray(config.get(cfgsec, 'pmax'))
    params = array[0]              # The initial guess

  indparams = mu.parray(config.get(cfgsec, 'indparams'))
  if indparams != [] and isinstance(indparams[0], str):
    indparams = mu.readbin(indparams[0])

  # Number of fitting parameters:
  nfree = np.sum(stepsize > 0)
  ifree  = np.where(stepsize > 0)[0] 
  iprior = np.where(priorlow > 0)[0] 

  # Get modeling function:
  func   = mu.parray(config.get(cfgsec, 'func'))
  if type(func) in [list, tuple, np.ndarray]:
    if len(func) == 3:
      sys.path.append(func[2])
    exec('from %s import %s as func'%(func[1], func[0]))
  elif not callable(func):
    return

  # Number of iterations:
  if nprays == 0:
    nprays = ndata
    shifts = np.arange(1, ndata)
  else:
    shifts = np.random.randint(0, ndata, nprays-1)

  # Allocate space for results:
  allfits = np.zeros((nprays, nfree))

  # Fit model:
  fitargs = (params, func, data, uncert, indparams, stepsize, pmin, pmax,
             (prior-params)[iprior], priorlow[iprior], priorup[iprior])
  chisq, dummy = mf.modelfit(params[ifree], args=fitargs)
  # Evaluate best model:
  fargs = [params] + indparams
  bestmodel = func(*fargs)
  chifactor = np.sqrt(chisq/(ndata-nfree))
  # Get residuals:
  residuals = data - bestmodel
  sigma     = np.copy(uncert*chifactor)

  allfits[0] = params[ifree]

  for i in np.arange(nprays-1):
    # Permuted data:
    pbdata = np.copy(bestmodel + np.roll(residuals, shifts[i]))
    # Permuted weights:
    pbunc  = np.roll(sigma, shifts[i])
    # Fitting parameters:
    pbfit = np.copy(params)[ifree]
    # Fit model:
    fitargs = (params, func, pbdata, pbunc, indparams, stepsize, pmin, pmax,
             (prior-params)[iprior], priorlow[iprior], priorup[iprior])
    chisq, dummy = mf.modelfit(pbfit, args=fitargs)
    allfits[i+1] = pbfit

  if savefile is not None:
    pbfile = open(savefile, "w")
    pbfile.write("Prayer-bead uncertainties:\n")
    pbunc = np.std(allfits,0)
    for j in np.arange(nfree):
      pbfile.write("%s  "%str(pbunc[j]))
    pbfile.close()

  return allfits, residuals

