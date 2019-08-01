# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['fit']

import numpy as np
import scipy.optimize as so

from . import stats as ms
from . import utils as mu


def fit(data, uncert, func, params, indparams=[],
        pstep=None, pmin=None, pmax=None,
        prior=None, priorlow=None, priorup=None, leastsq='lm'):
  """
  Find the best fitting params values using the Levenberg-Marquardt
  algorithm (wrapper of scipy.optimize.leastsq) considering shared and
  fixed parameters, and parameter Gaussian priors.

  This code minimizes the chi-square statistics:
    chisq = sum_i ((data[i]   - model[i])/uncert[i]     )**2.0 +
            sum_j ((params[j] - prior[j])/prioruncert[j])**2.0

  Parameters
  ----------
  data: 1D ndarray
      Dependent data fitted by func.
  uncert: 1D ndarray
      1-sigma uncertainty of data.
  func: callable
      The fitting function to model the data. It must be callable as:
      model = func(params, *indparams)
  params: 1D ndarray
      The model parameters.
  indparams: tuple
      Additional arguments required by func (if required).
  pstep: 1D ndarray
      Parameters' jump scale (same size as params).
      If the pstep is positive, the parameter is free for fitting.
      If the pstep is 0, keep the parameter value fixed.
      If the pstep is a negative integer, copy (share) the parameter value
      from params[np.abs(pstep)+1], which can be free or fixed.
  pmin: 1D ndarray
      Model parameters' lower boundaries (same size as params).
      Default -np.inf.
  pmax: 1D ndarray
      Model parameters' upper boundaries (same size as params).
      Default +np.inf.
  prior: 1D ndarray
      Model parameters' (Gaussian) prior values (same size as params).
      Considered only when priolow != 0.  priorlow and priorup are the
      lower and upper 1-sigma width of the Gaussian prior, respectively.
  priorlow: 1D ndarray
      Parameters' lower 1-sigma Gaussian prior (same size as params).
  priorup: 1D ndarray
      Paraneters' upper 1-sigma Gaussian prior (same size as params).
  leastsq: String
      Optimization algorithm:
      If 'lm': use the Levenberg-Marquardt algorithm
      If 'trf': use the Trust Region Reflective algorithm

  Returns
  -------
  mc3_output: Dict
      A dictionary containing the fit outputs, including:
      - chisq: Lowest chi-square value found by the optimizer.
      - bestp: Model parameters for the lowest chi-square value.
      - best_model: Model evaluated at for bestp.
      - optimizer_res: The output from the scipy optimizer.
  """
  with mu.Log() as log:
      if leastsq not in [None, 'lm', 'trf']:
          log.error("Invalid 'leastsq' input ({}). Must select from "
                    "['lm', 'trf'].".format(leastsq))

  # Total number of model parameters:
  npars = len(params)
  # Default pstep:
  if pstep is None:
      pstep = np.ones(npars, np.double)
  # Default boundaries (all parameter space):
  if pmin is None:
      pmin = np.tile(-np.inf, npars)
  if pmax is None:
      pmax = np.tile(np.inf,  npars)
  # Default priors, must set all or no one:
  if prior is None or priorlow is None or priorup is None:
      prior = priorup = priorlow = np.zeros(npars)

  # Cast to ndarrays:
  params   = np.asarray(params)
  pstep    = np.asarray(pstep)
  pmin     = np.asarray(pmin)
  pmax     = np.asarray(pmax)
  prior    = np.asarray(prior)
  priorlow = np.asarray(priorlow)
  priorup  = np.asarray(priorup)

  # Get indices:
  ifree  = np.where(pstep > 0)[0]
  ishare = np.where(pstep < 0)[0]

  fitparams = params[ifree]

  args = (params, func, data, uncert, indparams, pstep,
          prior, priorlow, priorup, ifree, ishare)
  # Levenberg-Marquardt optimization:
  if leastsq == 'lm':
      lsfit = so.leastsq(residuals, fitparams, args=args,
          ftol=3e-16, xtol=3e-16, gtol=3e-16, full_output=True)
      output, cov_x, infodict, mesg, err = lsfit
      params[ifree] = lsfit[0]
      resid = lsfit[2]["fvec"]

  # Bounded optimization:
  elif leastsq == 'trf':
      lsfit = so.least_squares(residuals, fitparams,
          bounds=(pmin[ifree], pmax[ifree]), args=args,
          ftol=3e-16, xtol=3e-16, gtol=3e-16, method='trf')
      params[ifree] = lsfit["x"]
      resid = lsfit["fun"]

  # Update shared parameters:
  for s in ishare:
      params[s] = params[-int(pstep[s])-1]

  # Compute best-fit model:
  bestmodel = func(params, *indparams)
  # Calculate chi-squared for best-fitting values:
  chisq = np.sum(resid**2.0)

  return {
      'chisq':chisq,
      'bestp': params,
      'best_model':bestmodel,
      'optimizer_res':lsfit,
  }


def residuals(fitparams, params, func, data, uncert, indparams, pstep,
              prior, priorlow, priorup, ifree, ishare):
  """
  Calculate the weighted residuals between data and a model, accounting
  also for parameter priors.

  Parameters
  ----------
  fitparams: 1D ndarray
      The model free parameters.
  params: 1D ndarray
      The model parameters (including fixed and shared parameters).
  func: Callable
      The fitting function to model the data, called as:
      model = func(params, *indparams)
  data: 1D ndarray
      Dependent data fitted by func.
  uncert: 1D ndarray
      1-sigma uncertainty of data.
  indparams: tuple
      Additional arguments required by func (if required).
  pstep: 1D ndarray
      Parameters' jump scale (same size as params).
      If the pstep is positive, the parameter is free for fitting.
      If the pstep is 0, keep the parameter value fixed.
      If the pstep is a negative integer, copy (share) the parameter value
      from params[np.abs(pstep)+1], which can be free or fixed.
  prior: 1D ndarray
      Model parameters' (Gaussian) prior values (same size as params).
      Considered only when priolow != 0.  priorlow and priorup are the
      lower and upper 1-sigma width of the Gaussian prior, respectively.
  priorlow: 1D ndarray
      Parameters' lower 1-sigma Gaussian prior (same size as params).
  priorup: 1D ndarray
      Paraneters' upper 1-sigma Gaussian prior (same size as params).
  ifree: 1D bool ndarray
      Indices of the free parameters in params.
  ishare: 1D bool ndarray
      Indices of the shared parameters in params.

  Returns
  -------
  Array of weighted data-model and prior-params residuals.
  """
  # Update params with fitparams:
  params[ifree] = fitparams

  # Update shared parameters:
  for s in ishare:
      params[s] = params[-int(pstep[s])-1]

  # Compute model:
  model = func(params, *indparams)
  # Calculate residuals:
  residuals = ms.residuals(model, data, uncert, params, prior,
                           priorlow, priorup)
  return residuals
