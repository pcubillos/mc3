# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'sample',
]

import os
import sys
import importlib
import multiprocessing as mpr
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

from .fit_driver import fit
from .mcmc_driver import mcmc
from . import utils as mu
from . import stats as ms
from . import plots as mp
from .version import __version__


def sample(
    data=None, uncert=None, func=None, params=None,
    indparams=[], indparams_dict={},
    pmin=None, pmax=None, pstep=None,
    prior=None, priorlow=None, priorup=None,
    sampler=None, ncpu=None, leastsq=None, chisqscale=False,
    nchains=7, nsamples=None, burnin=0, thinning=1,
    grtest=True, grbreak=0.0, grnmin=0.5, wlike=False,
    fgamma=1.0, fepsilon=0.0, hsize=10, kickoff='normal',
    plots=False, theme='blue', statistics='med_central',
    ioff=False, showbp=True,
    savefile=None, resume=False,
    rms=False, log=None, pnames=None, texnames=None,
    **kwargs):
    """
    This beautiful piece of code executes an MCMC or NS posterior sampling.

    Parameters
    ----------
    data: 1D float ndarray or string
        Data to be fit by func.  If string, path to file containing data.
    uncert: 1D float ndarray
        Uncertainties of data.
    func: Callable or string-iterable
        The callable function that models data as:
            model = func(params, *indparams, **indparams_dict)
        Or an iterable of 3 strings (funcname, modulename, path)
        that specifies the function name, function module, and module path.
        If the module is already in the python-path scope, path can be omitted.
    params: 1D float ndarray or string
        Set of initial fitting parameters for func.
        If string, path to file containing data.
    indparams: tuple or string
        Additional arguments required by func.  If string, path to file
        containing indparams.
    indparams_dict: dict
        Additional keyword arguments required by func (if needed).
    pmin: 1D ndarray
        Lower boundaries for the posterior exploration.
    pmax: 1D ndarray
        Upper boundaries for the posterior exploration.
    pstep: 1D ndarray
        Parameter stepping behavior.
        - Free parameters have pstep>0.
        - Fixed parameters have pstep=0.
        - Negative values indicate a shared parameter, with pstep set to
          the negative index of the sharing parameter (starting the count
          from 1), e.g.: to share second parameter and first one, do:
          pstep[1] = -1.
        For MCMC, the pstep value of free parameters set the scale of the
        initial jump proposal.
    prior: 1D ndarray
        Parameter priors.  The type of prior is determined by priorlow
        and priorup:
            if both priorlow>0 and priorup>0   Gaussian
            else                               Uniform between [pmin,pmax]
    priorlow: 1D ndarray
        Lower prior uncertainty values.
    priorup: 1D ndarray
        Upper prior uncertainty values.
    sampler: String
        Sampling algorithm:
        - 'mrw':  Metropolis random walk.
        - 'demc': Differential Evolution Markov chain.
        - 'snooker': DEMC-z with snooker update.
    ncpu: Integer
        Number of processors for the MCMC chains (mc3 defaults to
        one CPU for each chain plus a CPU for the central hub).
    leastsq: String
        If not None, perform a least-square optimization before the MCMC run.
        Select from:
            'lm': Levenberg-Marquardt (most efficient, but doesn't obey bounds)
            'trf': Trust Region Reflective
    chisqscale: Boolean
        Scale the data uncertainties such that the reduced chi-square = 1.
    nchains: Scalar
        Number of simultaneous chains to run.
    nsamples: Scalar
        Total number of samples.
    burnin: Integer
        Number of burned-in (discarded) number of iterations at the beginning
        of the chains.
    thinning: Integer
        Thinning factor of the chains (use every thinning-th iteration) used
        in the GR test and plots.
    wlike: Bool
        If True, calculate the likelihood in a wavelet-base.  This requires
        three additional parameters (TBD: this needs documentation).
    grtest: Boolean
        If True, run Gelman & Rubin test.
    grbreak: Float
        Gelman-Rubin convergence threshold to stop the MCMC (I'd suggest
        grbreak ~ 1.01).  Do not break if grbreak=0.0 (default).
    grnmin: Integer or float
        Minimum number of samples required for grbreak to stop the MCMC.
        If grnmin > 1: grnmin sets the minimum required number of samples.
        If 0 < grnmin < 1: grnmin sets the minimum required nsamples fraction.
    fgamma: Float
        Proposals jump scale factor for DEMC's gamma.
        The code computes: gamma = fgamma * 2.38 / sqrt(2*Nfree)
    fepsilon: Float
        Jump scale factor for DEMC's support distribution.
        The code computes: e = fepsilon * Normal(0, pstep)
    hsize: Integer
        Number of initial samples per chain.
    kickoff: String
        Flag to indicate how to start the chains:
        'normal' for normal distribution around initial guess, or
        'uniform' for uniform distribution withing the given boundaries.
    plots: Bool
        If True plot parameter traces, pairwise-posteriors, and posterior
        histograms.
    theme:
        The color theme for plots. Can have any format recognized as a
        matplotlib color.
    statistics: String
        Statistics to adopt for the plots. Select from:
        - 'med_central' Median and central quantile
        - 'max_like' Max marginal likelihood (mode) and HPD
        - 'global_max_like' Max a posteriori (best-fit) and HPD
    ioff: Bool
        If True, set plt.ioff(), i.e., do not display figures on screen.
    showbp: Bool
        If True, show best-fitting values in histogram and pairwise plots.
    savefile: String
        If not None, filename to store allparams and other MCMC results.
    resume: Boolean
        If True resume a previous run (identified by the .npz file name).
    rms: Boolean
        If True, calculate the RMS of the residuals: data - best_model.
    log: String or mc3.utils.Log instance
        Filename (as string) or log handler (as Log instance) handle logging.
    pnames: 1D string iterable
        List of parameter names (including fixed and shared parameters)
        to display on output screen and figures.  See also texnames.
        Screen output trims up to the 11th character.
        If not defined, default to texnames.
    texnames: 1D string iterable
        Parameter names for figures, which may use latex syntax.
        If not defined, default to pnames.
    kwargs: Dict
        Additional keyword arguments passed to the sampler.

    Returns
    -------
    mc3_output: Dict
        A Dictionary containing the MCMC posterior distribution and related
        stats, including:
        - posterior: thinned posterior distribution of shape [nsamples, nfree],
              including the burn-in phase.
        - zchain: chain indices for the posterior samples.
        - zmask: posterior mask to remove the burn-in.
        - chisq: chi^2 values for the posterior samples.
        - log_post: log(posterior) for the posterior samples (see Notes).
        - burnin: number of burned-in samples per chain.
        - ifree: Indices of the free parameters.
        - pnames: Parameter names.
        - texnames: Parameter names in Latex format.
        - meanp: mean of the marginal posteriors.
        - stdp: standard deviation of the marginal posteriors.
        - CRlo: lower boundary of the marginal 68%-highest posterior
              density (the credible region).
        - CRhi: upper boundary of the marginal 68%-HPD.
        - bestp: model parameters for the optimal log(posterior) in the sample.
        - best_log_post: optimal log(posterior) in the sample (see Notes).
        - best_model: model evaluated at bestp.
        - best_chisq: chi^2 for the optimal log(posterior) in the sample.
        - red_chisq: reduced chi-square: chi^2/(ndata-nfree) for the
              best-fitting sample.
        - BIC: Bayesian Information Criterion: chi^2 - nfree*log(ndata)
              for the best-fitting sample.
        - chisq_factor: Uncertainties scale factor to enforce chi^2_red = 1.
        - stddev_residuals: standard deviation of the residuals.
        - acceptance_rate: sample's acceptance rate.

    Notes
    -----
    The log_post variable is defined here as:
        log_post = log(posterior)
                 = log(likelihood) + log(prior)
                 = -0.5*chi-square + log_prior
                 = sum_i -0.5*((data[i] - model[i])/uncert[i])**2 + log_prior

    with log_prior defined as:
        log_prior = sum_j -0.5*((params[j] - prior[j])/prior_uncert[j])**2
    For each parameter with a Gaussian prior.
    Note that constant terms have been neglected.

    Examples
    --------
    >>> import numpy as np
    >>> import mc3

    >>> def quad(p, x):
    >>>     return p[0] + p[1]*x + p[2]*x**2.0

    >>> # Preamble, create a noisy synthetic dataset:
    >>> np.random.seed(3)
    >>> x = np.linspace(0, 10, 100)
    >>> p_true = [3, -2.4, 0.5]
    >>> y = quad(p_true, x)
    >>> uncert = np.sqrt(np.abs(y))
    >>> data = y + np.random.normal(0, uncert)

    >>> # Initial guess for fitting parameters:
    >>> params = np.array([ 3.0, -2.0,  0.1])
    >>> pstep  = np.array([ 1.0,  1.0,  1.0])
    >>> pmin   = np.array([ 0.0, -5.0, -1.0])
    >>> pmax   = np.array([10.0,  5.0,  1.0])

    >>> # Gaussian prior on first parameter, uniform on second and third:
    >>> prior    = np.array([3.5, 0.0, 0.0])
    >>> priorlow = np.array([0.1, 0.0, 0.0])
    >>> priorup  = np.array([0.1, 0.0, 0.0])

    >>> indparams = [x]
    >>> func = quad
    >>> ncpu = 7

    >>> # MCMC sampling:
    >>> mcmc_output = mc3.sample(
    >>>     data, uncert, func, params, indparams=indparams,
    >>>     sampler='snooker', pstep=pstep, ncpu=ncpu, pmin=pmin, pmax=pmax,
    >>>     prior=prior, priorlow=priorlow, priorup=priorup,
    >>>     leastsq='lm', nsamples=1e5, burnin=1000, plots=True)

    >>> # See more examples and details at:
    >>> # https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html
    """
    # Logging object:
    if isinstance(log, str):
        log = mu.Log(log, append=resume)
        closelog = True
    else:
        closelog = False
        if log is None:
            log = mu.Log()

    log.msg(
       f"\n{log.sep}\n"
        "  Multi-core Markov-chain Monte Carlo (mc3).\n"
       f"  Version {__version__}.\n"
       f"  Copyright (c) 2015-{date.today().year} Patricio Cubillos "
          "and collaborators.\n"
        "  mc3 is open-source software under the MIT license (see LICENSE).\n"
       f"{log.sep}\n\n")

    if sampler is None:
        log.error("'sampler' is a required argument")
    if nsamples is None and sampler in ['MRW', 'DEMC', 'snooker']:
        log.error("'nsamples' is a required argument for MCMC runs")
    if leastsq not in [None, 'lm', 'trf']:
        log.error(
            f"Invalid 'leastsq' input ({leastsq}). Must select from "
             "['lm', 'trf']")

    # Read the model parameters:
    params = mu.isfile(params, 'params', log, 'ascii', False, not_none=True)
    # Unpack if necessary:
    if np.ndim(params) > 1:
        ninfo, ndata = np.shape(params)
        if ninfo == 7:         # The priors
            prior = params[4]
            priorlow = params[5]
            priorup = params[6]
        if ninfo >= 4:         # The stepsize
            pstep = params[3]
        if ninfo >= 3:         # The boundaries
            pmin = params[1]
            pmax = params[2]
        else:
            log.error('Invalid format/shape for params input file')
        params = params[0]     # The initial guess
    params = np.array(params)

    # Process data and uncertainties:
    data = mu.isfile(data, 'data', log, 'bin', False, not_none=True)
    if np.ndim(data) > 1:
        data, uncert = data
    # Make local 'uncert' a copy, to avoid overwriting:
    if uncert is None:
        log.error("'uncert' is a required argument")
    uncert = np.copy(uncert)

    # Process the independent parameters:
    if indparams != []:
        indparams = mu.isfile(indparams, 'indparams', log, 'bin', unpack=False)

    if ioff:
        plt.ioff()

    resume = resume and (savefile is not None)
    if resume:
        log.msg(f"\n\n{log.sep}\n{log.sep}  Resuming previous MCMC run.\n\n")

    # Import the model function:
    if isinstance(func, (list, tuple, np.ndarray)):
        if len(func) == 3:
            sys.path.append(func[2])
        else:
            sys.path.append(os.getcwd())
        fmodule = importlib.import_module(func[1])
        func = getattr(fmodule, func[0])
    elif not callable(func):
        log.error(
            "'func' must be either a callable or an iterable of strings "
            "with the model function, file, and path names")

    if ncpu is None and sampler in ['snooker', 'demc', 'mrw']:
        ncpu = nchains
    # Cap the number of processors:
    if ncpu >= mpr.cpu_count():
        log.warning(
            f"The number of requested CPUs ({ncpu}) is >= than the number "
            f"of available CPUs ({mpr.cpu_count()}).  "
            f"Enforced ncpu to {mpr.cpu_count()-1}.")
        ncpu = mpr.cpu_count() - 1

    nparams = len(params)
    ndata = len(data)

    # Setup array of parameter names:
    if pnames is None and texnames is not None:
        pnames = texnames
    elif pnames is not None and texnames is None:
        texnames = pnames
    elif pnames is None and texnames is None:
        pnames = texnames = mu.default_parnames(nparams)
    pnames = np.asarray(pnames)
    texnames = np.asarray(texnames)

    if pmin is None:
        pmin = np.tile(-np.inf, nparams)
    if pmax is None:
        pmax = np.tile( np.inf, nparams)
    pmin = np.asarray(pmin)
    pmax = np.asarray(pmax)
    if pstep is None:
        pstep = 0.1 * np.abs(params)
    pstep = np.asarray(pstep)

    # Set prior parameter indices:
    if prior is None or priorup is None or priorlow is None:
        prior = priorup = priorlow = np.zeros(nparams)

    # Override priors for non-free parameters:
    priorlow[pstep<=0] = 0.0
    priorup [pstep<=0] = 0.0

    # Check that initial values lie within the boundaries:
    if np.any(params < pmin) or np.any(params > pmax):
        pout = ""
        for pname, par, minp, maxp in zip(pnames, params, pmin, pmax):
            if par < minp:
                pout += f"\n{pname[:11]:11s}  {minp: 12.5e} < {par: 12.5e}"
            if par > maxp:
                pout += f"\n{pname[:11]:26s}  {par: 12.5e} > {maxp: 12.5e}"

        log.error(
            "Some initial-guess values are out of bounds:\n"
            "Param name           pmin          value           pmax\n"
            "-----------  ------------   ------------   ------------"
            f"{pout}"
        )

    nfree = int(np.sum(pstep > 0))
    ifree = np.where(pstep > 0)[0]  # Free parameter indices
    ishare = np.where(pstep < 0)[0]  # Shared parameter indices

    # Check output dimension:
    model0 = func(params, *indparams, **indparams_dict)
    if np.shape(model0) != np.shape(data):
        log.error(
            f"The size of the data array ({np.size(data)}) does not "
            f"match the size of the func() output ({np.size(model0)})"
        )

    # Check that output path exists:
    if savefile is not None:
        fpath, fname = os.path.split(os.path.realpath(savefile))
        if not os.path.exists(fpath):
            log.warning(
                f"Output folder path: '{fpath}' does not exist. "
                "Creating new folder."
            )
            os.makedirs(fpath)

    # Least-squares minimization:
    chisq_factor = 1.0
    if leastsq is not None:
        fit_output = fit(
            data, uncert, func, np.copy(params),
            indparams, indparams_dict,
            pstep, pmin, pmax, prior, priorlow, priorup, leastsq)
        fit_bestp = fit_output['bestp']
        log.msg(
            f"Least-squares best-fitting parameters:\n  {fit_bestp}\n\n",
            si=2)

        # Scale data-uncertainties such that reduced chisq = 1:
        if chisqscale:
            chisq_factor = np.sqrt(fit_output['best_chisq']/(ndata-nfree))
            uncert *= chisq_factor

            # Re-calculate best-fitting parameters with new uncertainties:
            fit_output = fit(
                data, uncert, func, np.copy(params),
                indparams, indparams_dict,
                pstep, pmin, pmax, prior, priorlow, priorup, leastsq)
            log.msg(
                "Least-squares best-fitting parameters (rescaled chisq):"
                f"\n  {fit_output['bestp']}\n\n",
                si=2)
        params = np.copy(fit_output['bestp'])
    else:
        fit_output = None

    if resume:
        with np.load(savefile) as oldrun:
            uncert *= float(oldrun['chisq_factor'])/chisq_factor
            chisq_factor = float(oldrun['chisq_factor'])

    # Here's where the magic happens:
    if sampler in ['mrw', 'demc', 'snooker']:
        output = mcmc(
            data, uncert, func,
            params, indparams, indparams_dict,
            pmin, pmax, pstep,
            prior, priorlow, priorup, nchains, ncpu, nsamples, sampler,
            wlike, fit_output, grtest, grbreak, grnmin, burnin, thinning,
            fgamma, fepsilon, hsize, kickoff, savefile, resume, log,
            pnames, texnames,
        )

    # Get some stats:
    output['chisq_factor'] = chisq_factor

    if leastsq is not None:
        delta_log_post = output['best_log_post'] - fit_output['best_log_post']
        delta_pars = output['bestp'] - fit_output['bestp']
        if delta_log_post > 5.0e-8 and np.any(delta_pars != 0.0):
            log.warning(
                "MCMC found a better fit than the minimizer:\n"
                "MCMC best-fitting parameters:        (chisq={:.8g})\n{}\n"
                "Minimizer best-fitting parameters:   (chisq={:.8g})\n{}".
                format(
                    -2*output['best_log_post'], output['bestp'],
                    -2*fit_output['best_log_post'], fit_output['bestp']))

    # Stats without burn-in samples:
    posterior, zchain, zmask = mu.burn(
        Z=output['posterior'], zchain=output['zchain'], burnin=output['burnin'])

    bestp = output['bestp']
    post = mp.Posterior(
        posterior, pnames=texnames[ifree], theme=theme,
        bestp=bestp[ifree], statistics=statistics,
    )
    # Let Posterior to turn the theme into a Theme() object:
    theme = post.theme

    # Parameter statistics:
    sample_stats = ms.calc_sample_statistics(
        post.posterior, bestp, pstep, calc_hpd=True,
    )
    median = output['medianp'] = sample_stats[0]
    mean = output['meanp'] = sample_stats[1]
    stdp = output['stdp'] = sample_stats[2]
    med_low_bounds = output['median_low_bounds'] = sample_stats[3]
    med_high_bounds = output['median_high_bounds'] = sample_stats[4]
    mode = output['mode'] = sample_stats[5]
    hpd_low_bounds = output['hpd_low_bounds'] = sample_stats[6]
    hpd_high_bounds = output['hpd_high_bounds'] = sample_stats[7]
    # Legacy (this will be deprecated at some point)
    output['CRlo'] = hpd_low_bounds - bestp
    output['CRhi'] = hpd_high_bounds - bestp
    output['CRlo'][pstep==0] = output['CRhi'][pstep==0] = 0.0

    log.msg(
        "\nParameter name     best fit   median      1sigma_low   1sigma_hi        S/N"
        "\n--------------- -----------  -----------------------------------  ---------",
        width=80)
    for i in range(nparams):
        pname = f'{pnames[i][0:15]:<15}'
        lo = med_low_bounds[i] - median[i]
        hi = med_high_bounds[i] - median[i]
        if i in ifree:
            snr = f"{np.abs(bestp[i])/stdp[i]:.1f}"
        elif i in ishare:
            idx = -int(pstep[i])
            snr = f"[share{idx:02d}]"
        else:
            snr = "[fixed]"
            lo = hi = 0.0
        log.msg(
            f"{pname} {bestp[i]:11.4e}  {median[i]:11.4e} "
            f"{lo:11.4e} {hi:11.4e}  {snr:>9s}",
            width=160,
        )

    # Fit statistics:
    best_chisq = output['best_chisq']
    log_post = -2.0*output['best_log_post']
    bic = output['BIC']
    red_chisq = output['red_chisq']
    std_dev = output['stddev_residuals']

    chisqscale_txt = f"sqrt(reduced chi-squared) factor: {chisq_factor:.4f}\n"
    if not chisqscale:
        chisqscale_txt = ''

    fmt = len(f"{bic:.4f}")  # Length of string formatting
    log.msg(
        f"\n{chisqscale_txt}"
        f"Best-parameter's chi-squared:       {best_chisq:{fmt}.4f}\n"
        f"Best-parameter's -2*log(posterior): {log_post:{fmt}.4f}\n"
        f"Bayesian Information Criterion:     {bic:{fmt}.4f}\n"
        f"Reduced chi-squared:                {red_chisq:{fmt}.4f}\n"
        f"Standard deviation of residuals:  {std_dev:.6g}\n",
        indent=2,
    )

    # Extract filename from savefile or default one:
    if savefile is not None:
        savefile_root = os.path.splitext(savefile)[0]
    else:
        savefile_root = 'mc3'

    stats_file = f'{savefile_root}_statistics.txt'
    ms.summary_stats(post, output, filename=stats_file)
    log.msg(
        '\nFor a detailed summary with all parameter posterior statistics '
        f'see {stats_file}',
    )

    log.msg("\nOutput sampler files:")
    log.msg(stats_file, indent=2)

    if savefile is not None:
        np.savez(savefile, **output)
        log.msg(savefile, indent=2)

    if plots:
        # Trace plot:
        savefile = f'{savefile_root}_trace.png'
        mp.trace(
            output['posterior'], zchain=output['zchain'],
            burnin=output['burnin'], pnames=texnames[ifree],
            savefile=savefile, color=theme.color,
        )
        log.msg(savefile, indent=2)
        # Pairwise posteriors:
        savefile = f'{savefile_root}_pairwise_posterior.png'
        post.plot(savefile=savefile, show_estimates=showbp)
        log.msg(savefile, indent=2)
        # Histograms:
        savefile = f'{savefile_root}_marginal_posterior.png'
        post.plot_histogram(savefile=savefile, show_estimates=showbp)
        log.msg(savefile, indent=2)
        # RMS vs bin size:
        if rms:
            savefile = f'{savefile_root}_RMS.png'
            residuals = output['best_model'] - data
            data_rms, rms_lo, rms_hi, stderr, binsize = ms.time_avg(residuals)
            mp.rms(
                binsize, data_rms, stderr, rms_lo, rms_hi,
                binstep=len(binsize)//500+1,
                savefile=savefile,
            )
            log.msg(savefile, indent=2)

    # Close the log file if necessary:
    if closelog:
        log.msg(log.logname, indent=2)
        log.close()

    return output
