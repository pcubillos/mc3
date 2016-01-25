# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from __future__ import absolute_import
import sys, os, ConfigParser
import numpy   as np

from .. import utils as mu
from .. import fit   as mf

def prayer(configfile, nprays=0, savefile=None):
  """
  Implement a Prayer-bead method to estimate parameter uncertainties.

  Parameters
  ----------
  configfile: String
    Configuration file name
  nprays: Integer
    Number of prayer-bead shifts.  If nprays==0, set to the number
    of data points.
  savefile: String
    Name of file where to store the prayer-bead results.

  Notes
  -----
  Believing in a prayer bead is a mere act of faith, we are scientists
  for god's sake!
  """

  config = ConfigParser.SafeConfigParser()
  config.read([configfile])
  cfgsec = "MCMC" 

  data = mu.parray(config.get(cfgsec, 'data'))
  if isinstance(data[0], str):
    array = mu.loadbin(data[0])
    data = array[0]
    if len(array) == 2:
      uncert = array[1]
    else:
      uncert = mu.parray(config.get(cfgsec, 'uncert'))

  params    = mu.parray(config.get(cfgsec, 'params'))
  if isinstance(params[0], str):
    array = mu.loadascii(params[0])
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
    indparams = mu.loadbin(indparams[0])

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
    pbunc = np.std(allfits, 0)
    for j in np.arange(nfree):
      pbfile.write("%s  "%str(pbunc[j]))
    pbfile.close()

  return allfits, residuals

