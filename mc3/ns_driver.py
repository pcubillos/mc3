# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["nested_sampling"]

import sys
import numpy as np
from . import stats as ms

if sys.version_info.major == 2:
    range = xrange


def resample_equal(weights, rstate=None):
    """Based on dynesty.utils.resample_equal()"""
    SQRTEPS = np.sqrt(float(np.finfo(np.float64).eps))
    if rstate is None:
        rstate = np.random
    if abs(np.sum(weights) - 1.) > SQRTEPS:
        raise ValueError("Weights do not sum to 1.")

    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    idx = np.zeros(nsamples, dtype=np.int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def nested_sampling(data, uncert, func, params, indparams, pmin, pmax, pstep,
    prior, priorlow, priorup, ncpu, thinning, resume, log,
    **kwargs):
  """
  This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

  Parameters
  ----------
  data: 1D float ndarray
      Data to be fit by func.
  uncert: 1D float ndarray
      Uncertainties of data.
  func: Callable
      The callable function that models data as model=func(params,*indparams)
  params: 1D float ndarray
      Set of initial fitting parameters for func.
  indparams: tuple
      Additional arguments required by func.
  pmin: 1D ndarray
      Lower boundaries for the posterior exploration.
  pmax: 1D ndarray
      Upper boundaries for the posterior exploration.
  pstep: 1D ndarray
      Parameter stepping.
  prior: 1D ndarray
      Parameter prior distribution means.
  priorlow: 1D ndarray
      Lower prior uncertainty values.
  priorup: 1D ndarray
      Upper prior uncertainty values.
  ncpu: Integer
      Number of processors for the NS sampling.
  thinning: Integer
      Thinning factor of the samples.
  resume: Boolean
      If True resume a previous run.
  log: mc3.utils.Log instance
      Logging object.

  Returns
  -------
  output: Dict
      A Dictionary containing the NS posterior distribution and related
      stats, including:
      - Z: thinned posterior distribution of shape [nsamples, nfree].
      - Zchain: chain indices for each sample in Z.
      - chisq: chi^2 value for each sample in Z.
      - Zlog_post: chi^2 + sum()
      - burnin: number of burned-in samples per chain (zero for NS).
      - bestp: model parameters for the lowest-chi^2 sample.
      - best_chisq: lowest-chi^2 in the sample.
      - best_model: model evaluated at bestp.
      - eff: sampling efficiency.

  Examples
  --------
  >>> # See example in mc3.sample() and
  >>> # https://mc3.readthedocs.io/en/latest/ns_tutorial.html
  """
  try:
      import dynesty
      if ncpu > 1:
          from pathos.multiprocessing import ProcessingPool
  except ImportError as error:
      log.error("ModuleNotFoundError: {}".format(error))

  nfree  = int(np.sum(pstep > 0))
  ifree  = np.where(pstep > 0)[0]
  ishare = np.where(pstep < 0)[0]
  # Can't use multiprocessing.Pool since map can't pickle defs:
  pool = ProcessingPool(nodes=ncpu) if ncpu > 1 else None
  queue_size = ncpu if ncpu > 1 else None

  # Setup prior transform:
  priors = []
  for p0, plo, pup, min, max, step in zip(prior, priorlow, priorup,
          pmin, pmax, pstep):
      if step <= 0:
          continue
      if plo == 0.0 and pup == 0.0:
          priors.append(ms.ppf_uniform(min, max))
      else:
          priors.append(ms.ppf_gaussian(p0, plo, pup))

  def prior_transform(u):
      return [p(v) for p,v in zip(priors,u)]

  def loglike(pars):
      params[ifree] = pars
      for s in ishare:
          params[s] = params[-int(pstep[s])-1]
      return -0.5*np.sum((data-func(params, *indparams))**2/uncert**2)

  # Intercept kwargs that go into DynamicNestedSampler():
  skip_logp = False
  if 'loglikelihood' in kwargs:
      loglike = kwargs.pop('loglikelihood')
  if 'prior_transform' in kwargs:
      prior_transform = kwargs.pop('prior_transform')
      skip_logp = True
  if 'ndim' in kwargs:
      nfree = kwargs.pop('ndim')
  if 'pool' in kwargs:
      pool = kwargs.pop('pool')
  if 'queue_size' in kwargs:
      queue_size = kwargs.pop('queue_size')

  # Run dynesty:
  sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, nfree,
      pool=pool, queue_size=queue_size)
  sampler.run_nested(**kwargs)

  weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
  isample = resample_equal(weights)
  posterior = sampler.results.samples[isample] #[::thinning]
  chisq = -2.0*sampler.results.logl[isample]

  # Contribution to chi-square from non-uniform priors:
  if skip_logp:
      # TBD: Compute from prior_transform()
      log_prior = 0.0
  else:
      log_prior = ms.log_prior(posterior, prior, priorlow, priorup, pstep)
  log_post = chisq + log_prior

  # Best-fit statistics from sample:
  ibest = np.argmin(log_post)
  bestp = np.copy(params)
  bestp[ifree] = posterior[ibest]
  for s in ishare:
      bestp[s] = bestp[-int(pstep[s])-1]
  best_model = func(bestp, *indparams)
  best_chisq = chisq[ibest]
  best_log_post = log_post[ibest]

  # Print out Summary:
  log.msg("\nNested Sampling Summary:"
          "\n------------------------")

  # Get some stats:
  nsample  = sampler.results['niter']
  nZsample = len(posterior)  # Valid samples (after thinning and burning)

  fmt = len(str(nsample))
  log.msg("Number of evaluated samples:  {:{}d}".
          format(nsample,  fmt), indent=2)
  log.msg("Thinning factor:              {:{}d}".
          format(thinning, fmt), indent=2)
  log.msg("NS sample size (thinned):     {:{}d}".
          format(nZsample, fmt), indent=2)
  log.msg("Sampling efficiency:  {:.2f}%\n".
          format(sampler.results['eff']), indent=2)

  # Build the output dict:
  output = {
      # The posterior:
      'posterior':posterior,
      'zchain':np.zeros(nsample, int),
      'chisq':chisq,
      'log_post':log_post,
      'burnin':0,
      # Posterior stats:
      'eff':sampler.results['eff'],
      # Best-fit stats:
      'bestp':bestp,
      'best_model':best_model,
      'best_log_post':best_log_post,
      'best_chisq':best_chisq,
      # Extra stuff:
      'dynesty_sampler':sampler,
  }
  return output

