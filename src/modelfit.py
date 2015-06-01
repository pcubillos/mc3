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

import sys, os
import numpy as np
import scipy.optimize as so

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/cfuncs/lib')
import chisq as cs

def modelfit(fitparams, args):
  """
  Find the best fitting fitparams values using the Levemberg-Mardquardt
  algorithm (wrapper of scipy's leastsq)

  Parameters:
  -----------
  fitparams: 1D ndarray
     The model fitting parameters to fit.
  args: Tuple
     Tuple of additional arguments passed to residuals function (see
     residuals docstring).

  Returns:
  --------
  chisq: Float
     Chi-squared for the best fitting values found.

  Modification History:
  ---------------------
  2014-05-09  patricio  Initial implementation for MC3.
                        pcubillos@fulbrightmail.org
  2014-06-09  patricio  Fixed glitch with informative priors.
  """
  # Call leastsq minimizer:
  fit = so.leastsq(residuals, fitparams, args=args, #maxfev=300,
                   ftol=1e-16, xtol=1e-16, gtol=1e-16, full_output=True)
  output, cov_x, infodict, mesg, err = fit

  # Calculate chi-squared:
  rargs = [output] + list(args)
  resid = residuals(*rargs)
  chisq = np.sum(resid**2.0)
  return chisq, fit


def residuals(fitparams, params, func, data, uncert, indparams, stepsize,
              pmin, pmax, prior, priorlow, priorup):
  """
  Calculate the weighted residuals between data and a model, accounting
  also for parameter priors.

  Parameters:
  -----------
  fitparams: 1D ndarray
     The model free parameters.
  params: 1D ndarray
     Model parameters (including fixed and shared parameters).
  func: Callable
     Function that models data, being called as:
     model = func(params, *indparams)
  data: 1D ndarray
     The data set being modeled.
  uncert: 1D ndarray
     Data uncertainties.
  indparams: Tuple
     Additional arguments for the func function.
  stepsize: 1D ndarray
     Array indicating which params are free params (stepsize > 0), the
     fixed params (stepsize=0) and shared parameters (stepsize < 0).
  pmin: 1D ndarray
     Lower boundaries of params.
  pmax: 1D ndarray
     Upper boundaries of params.
  prior: 1D ndarray
     Priors array.
  priorlow: 1D ndarray
     Priors lower uncertainties.
  priorup: 1D ndarray
     Priors upper uncertainties.

  Returns:
  --------
  Array of weighted data-model and prior-params residuals.

  Modification History:
  ---------------------
  2014-05-09  patricio  Initial implementation for MC3.
                        pcubillos@fulbrightmail.org
  2014-06-09  patricio  Changed prioroff to prior (bug fix).
  """
  # Get free and shared indices:
  ifree  = np.where(stepsize > 0)[0]
  ishare = np.where(stepsize < 0)[0]

  # Combine fitparams into func params:
  params[ifree] = fitparams

  # Keep parameters within boundaries:
  params = np.clip(params, pmin, pmax)

  # Update shared parameters:
  for s in ishare:
    params[s] = params[-int(stepsize[s])-1]

  # Compute model:
  fargs = [params] + indparams
  model = func(*fargs)

  # Find the parameters that have prior:
  iprior = np.where(priorlow != 0)[0]
  prioroff = params - prior

  # Calculate residuals:
  residuals = cs.residuals(model, data, uncert,
                           prioroff[iprior], priorlow[iprior], priorup[iprior])
  #print("Params: %s"%str(params))
  #print("Prior:  %s"%str(prior))
  #print(prioroff[iprior], priorlow[iprior], priorup[iprior])
  #print(residuals[-4:])
  return residuals
