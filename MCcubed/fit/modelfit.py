# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["modelfit", "residuals"]

import sys, os
import numpy as np
import scipy.optimize as so

import chisq as cs

def modelfit(fitparams, args):
  """
  Find the best fitting fitparams values using the Levemberg-Mardquardt
  algorithm (wrapper of scipy's leastsq)

  Parameters
  ----------
  fitparams: 1D ndarray
     The model fitting parameters to fit.
  args: Tuple
     Tuple of additional arguments passed to residuals function (see
     residuals docstring).

  Returns
  -------
  chisq: Float
     Chi-squared for the best fitting values found.
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

  Parameters
  ----------
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

  Returns
  -------
  Array of weighted data-model and prior-params residuals.
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
