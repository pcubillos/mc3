# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ['fit']

import numpy as np
import scipy.optimize as so

from . import stats as ms
from . import utils as mu


def fit(data, uncert, func, params, indparams=[],
        pstep=None, pmin=None, pmax=None,
        prior=None, priorlow=None, priorup=None, leastsq='lm'):
  r"""
  Find the best-fitting params values to the dataset by performing a
  Maximum-A-Posteriori optimization.

  This is achieved by minimizing the negative log posterior, with:
  log_post = log(posterior)
           = log(likelihood) + log(prior)
           = -0.5*chi-squared + log_prior
           = sum_i -0.5*((data[i] - model[i])/uncert[i])**2 + log_prior

  where log_prior is defined as:
      log_prior = sum -0.5*((params - prior)/prior_uncert)**2
  for each parameter with a Gaussian prior; parameters with
  uniform priors do not contribute to log_prior.

  Constant terms have been neglected since they don't affect the
  optimization.

  Parameters
  ----------
  data: 1D ndarray
      Data fitted by func.
  uncert: 1D ndarray
      1-sigma uncertainties of data.
  func: callable
      The fitting function to model the data. It must be callable as:
      model = func(params, *indparams)
  params: 1D ndarray
      The model parameters.
  indparams: tuple
      Additional arguments required by func (if required).
  pstep: 1D ndarray
      Parameters fitting behavior.
      If pstep is positive, the parameter is free for fitting.
      If pstep is zero, keep the parameter value fixed.
      If pstep is a negative integer, copy the value from
          params[np.abs(pstep)+1].
  pmin: 1D ndarray
      Model parameters' lower boundaries.  Default -np.inf.
      Only for leastsq='trf', since 'lm' does not handle bounds.
  pmax: 1D ndarray
      Model parameters' upper boundaries.  Default +np.inf.
      Only for leastsq='trf', since 'lm' does not handle bounds.
  prior: 1D ndarray
      Parameters priors.  The type of prior is determined by priorlow
      and priorup:
          Gaussian: if both priorlow>0 and priorup>0
          Uniform:  else
  priorlow: 1D ndarray
      Parameters' lower 1-sigma Gaussian prior.
  priorup: 1D ndarray
      Paraneters' upper 1-sigma Gaussian prior.
  leastsq: String
      Optimization algorithm:
      If 'lm': use the Levenberg-Marquardt algorithm
      If 'trf': use the Trust Region Reflective algorithm

  Returns
  -------
  mc3_output: Dict
      A dictionary containing the fit outputs, including:
      - best_log_post: optimal log of the posterior (as defined above).
      - best_chisq: chi-square for the found best_log_post.
      - best_model: model evaluated at bestp.
      - bestp: Model parameters for the optimal best_log_post.
      - optimizer_res: the output from the scipy optimizer.

  Examples
  --------
  >>> import mc3
  >>> import numpy as np

  >>> def quad(p, x):
  >>>     '''Quadratic polynomial: y(x) = p0 + p1*x + p2*x^2'''
  >>>     return p[0] + p[1]*x + p[2]*x**2.0

  >>> # Preamble, create a noisy synthetic dataset:
  >>> np.random.seed(10)
  >>> x = np.linspace(0, 10, 100)
  >>> p_true = [4.5, -2.4, 0.5]
  >>> y = quad(p_true, x)
  >>> uncert = np.sqrt(np.abs(y))
  >>> data = y + np.random.normal(0, uncert)

  >>> # Initial guess for fitting parameters:
  >>> params = np.array([ 3.0, -2.0,  0.1])

  >>> # Fit data:
  >>> output = mc3.fit(data, uncert, quad, params, indparams=[x])
  >>> print(output['bestp'], output['best_chisq'], -2*output['best_log_post'], sep='\n')
  [ 4.57471072 -2.28357843  0.48341911]
  92.79923183159411
  92.79923183159411

  >>> # Fit with priors (Gaussian, uniform, uniform):
  >>> prior    = np.array([4.0, 0.0, 0.0])
  >>> priorlow = np.array([0.1, 0.0, 0.0])
  >>> priorup  = np.array([0.1, 0.0, 0.0])
  >>> output = mc3.fit(data, uncert, quad, params, indparams=[x],
          prior=prior, priorlow=priorlow, priorup=priorup)
  >>> print(output['bestp'], output['best_chisq'], -2*output['best_log_post'], sep='\n')
  [ 4.01743461 -2.00989433  0.45686521]
  93.77082119449915
  93.80121777303248
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
  best_model = func(params, *indparams)
  # Calculate chi-squared for best-fitting values:
  best_log_post = -0.5*np.sum(resid**2.0)
  log_prior = ms.log_prior(params[ifree], prior, priorlow, priorup, pstep)
  best_chisq = -2*(best_log_post - log_prior)

  return {
      'bestp':params,
      'best_log_post':best_log_post,
      'best_chisq':best_chisq,
      'best_model':best_model,
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
