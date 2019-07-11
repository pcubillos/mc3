API
===


MCcubed
_______


.. py:module:: MCcubed

.. py:function:: mcmc(data=None, uncert=None, func=None, indparams=[], params=None, pmin=None, pmax=None, pstep=None, prior=None, priorlow=None, priorup=None, nchains=7, ncpu=None, nsamples=100000.0, walk='snooker', wlike=False, leastsq=False, lm=False, chisqscale=False, grtest=True, grbreak=0.0, grnmin=0.5, burnin=0, thinning=1, fgamma=1.0, fepsilon=0.0, hsize=10, kickoff='normal', plots=False, ioff=False, showbp=True, savefile=None, savemodel=None, resume=False, rms=False, log=None, pnames=None, texnames=None, parname=None, nproc=None, stepsize=None, full_output=None, chireturn=None)
.. code-block:: pycon

    This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

    Parameters
    ----------
    data: 1D float ndarray or string
        Data to be fit by func.  If string, path to file containing data.
    uncert: 1D float ndarray
        Uncertainties of data.
    func: Callable or string-iterable
        The callable function that models data as:
            model = func(params, *indparams)
        Or an iterable of 3 strings (funcname, modulename, path)
        that specifies the function name, function module, and module path.
        If the module is already in the python-path scope, path can be omitted.
    indparams: tuple or string
        Additional arguments required by func.  If string, path to file
        containing indparams.
    params: 1D/2D float ndarray or string
        Set of initial fitting parameters for func.  If 2D, of shape
        (nparams, nchains), it is assumed that it is one set for each chain.
        If string, path to file containing data.
    pmin: 1D ndarray
       Lower boundaries for the posterior exploration.
    pmax: 1D ndarray
       Upper boundaries for the posterior exploration.
    pstep: 1D ndarray
       Parameter stepping.  If a value is 0, keep the parameter fixed.
       Negative values indicate a shared parameter (See Note 1).
    prior: 1D ndarray
       Parameter prior distribution means (See Note 2).
    priorlow: 1D ndarray
       Lower prior uncertainty values (See Note 2).
    priorup: 1D ndarray
       Upper prior uncertainty values (See Note 2).
    nchains: Scalar
       Number of simultaneous chains to run.
    ncpu: Integer
       Number of processors for the MCMC chains (MC3 defaults to
       one CPU for each chain plus a CPU for the central hub).
    nsamples: Scalar
       Total number of samples.
    walk: String
       Random walk algorithm:
       - 'mrw':  Metropolis random walk.
       - 'demc': Differential Evolution Markov chain.
       - 'snooker': DEMC-z with snooker update.
    wlike: Boolean
       If True, calculate the likelihood in a wavelet-base.  This requires
       three additional parameters (See Note 3).
    leastsq: Boolean
       Perform a least-square minimization before the MCMC run.
    lm: Boolean
       If True use the Levenberg-Marquardt algorithm for the optimization.
       If False, use the Trust Region Reflective algorithm.
    chisqscale: Boolean
       Scale the data uncertainties such that the reduced chi-squared = 1.
    grtest: Boolean
       Run Gelman & Rubin test.
    grbreak: Float
       Gelman-Rubin convergence threshold to stop the MCMC (I'd suggest
       grbreak ~ 1.001--1.005).  Do not break if grbreak=0.0 (default).
    grnmin: Integer or float
       Minimum number of samples required for grbreak to stop the MCMC.
       If grnmin > 1: grnmin sets the minimum required number of samples.
       If 0 < grnmin < 1: grnmin sets the minimum required nsamples fraction.
    burnin: Integer
       Number of burned-in (discarded) number of iterations at the beginning
       of the chains.
    thinning: Integer
       Thinning factor of the chains (use every thinning-th iteration) used
       in the GR test and plots.
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
    ioff: Bool
       If True, set plt.ioff(), i.e., do not display figures on screen.
    showbp: Bool
       If True, show best-fitting values in histogram and pairwise plots.
    savefile: String
       If not None, filename to store allparams and other MCMC results.
    savemodel: String
       If not None, filename to store the values of the evaluated function.
    resume: Boolean
       If True resume a previous run.
    rms: Boolean
       If True, calculate the RMS of the residuals: data - bestmodel.
    log: String or FILE pointer
       Filename or File object to write log.
    pnames: 1D string iterable
       List of parameter names (including fixed and shared parameters)
       to display on output screen and figures.  See also texnames.
       Screen output trims up to the 11th character.
       If not defined, default to texnames.
    texnames: 1D string iterable
       Parameter names for figures, which may use latex syntax.
       If not defined, default to pnames.
    parname: 1D string ndarray
        Deprecated, use pnames instead.
    nproc: Integer
        Deprecated, use ncpu instead.
    stepsize: 1D ndarray
        Deprecated, use pstep instead.
    chireturn:
        Deprecated.
    full_output:  Bool
        Deprecated.

    Returns
    -------
    bestp: 1D ndarray
       Array of the best-fitting parameters (including fixed and shared).
    CRlo:  1D ndarray
       The lower boundary of the marginal 68%-highest posterior density
       (the credible region) for each parameter, with respect to bestp.
    CRhi:  1D ndarray
       The upper boundary of the marginal 68%-highest posterior density
       (the credible region) for each parameter, with respect to bestp.
    stdp: 1D ndarray
       Array of the best-fitting parameter uncertainties, calculated as the
       standard deviation of the marginalized, thinned, burned-in posterior.
    posterior: 2D float ndarray
       An array of shape (Nfreepars, Nsamples) with the thinned MCMC posterior
       distribution of the fitting parameters (excluding fixed and shared).
       If full_output is True, the posterior includes the burnin samples.
    Zchain: 1D integer ndarray
       Index of the chain for each sample in posterior.  M0 samples have chain
       index of -1.
    chiout: 4-elements tuple
       Tuple containing the best-fit chi-square, reduced chi-square, scale
       factor to enforce redchisq=1, and the Bayesian information
       criterion (BIC).

    Notes
    -----
    1.- To set one parameter equal to another, set its pstep to the
        negative index in params (Starting the count from 1); e.g.: to set
        the second parameter equal to the first one, do: pstep[1] = -1.
    2.- If any of the fitting parameters has a prior estimate, e.g.,
          param[i] = p0 +up/-low,
        with up and low the 1sigma uncertainties.  This information can be
        considered in the MCMC run by setting:
        prior[i]    = p0
        priorup[i]  = up
        priorlow[i] = low
        All three: prior, priorup, and priorlow must be set and, furthermore,
        priorup and priorlow must be > 0 to be considered as prior.
    3.- If data, uncert, params, pmin, pmax, pstep, prior, priorlow,
        or priorup are set as filenames, the file must contain one value per
        line.
        For simplicity, the data file can hold both data and uncert arrays.
        In this case, each line contains one value from each array per line,
        separated by an empty-space character.
        Similarly, params can hold: params, pmin, pmax, pstep, priorlow,
        and priorup.  The file can hold as few or as many array as long as
        they are provided in that exact order.
    4.- An indparams file works differently, the file will be interpreted
        as a list of arguments, one in each line.  If there is more than one
        element per line (empty-space separated), it will be interpreted as
        an array.
    5.- FINDME: WAVELET LIKELIHOOD

    Examples
    --------
    >>> # See https://github.com/pcubillos/MCcubed/tree/master/examples


MCcubed.fit
___________


.. py:module:: MCcubed.fit

.. py:function:: modelfit(params, func, data, uncert, indparams=[], pstep=None, pmin=None, pmax=None, prior=None, priorlow=None, priorup=None, lm=False)
.. code-block:: pycon

    Find the best fitting params values using the Levenberg-Marquardt
    algorithm (wrapper of scipy.optimize.leastsq) considering shared and
    fixed parameters, and parameter Gaussian priors.

    This code minimizes the chi-square statistics:
      chisq = sum_i ((data[i]   - model[i])/uncert[i]     )**2.0 +
              sum_j ((params[j] - prior[j])/prioruncert[j])**2.0

    Parameters
    ----------
    params: 1D ndarray
       The model parameters.
    func: callable or string-iterable
       The fitting function to model the data as:
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
    lm: Bool
       If True use the Levenberg-Marquardt algorithm (through
       scipy.optimize.leastsq).  If False (default), use the Trust Region
       Reflective algorithm (through scipy.optimize.least_squares).

    Returns
    -------
    chisq: Float
       Chi-squared for the best fitting values.
    bestparams: 1D float ndarray
       Array of best-fitting parameters (including fixed and shared params).
    bestmodel: 1D float ndarray
       Evaluated model for bestparams.
    lsfit: List
       The output from the scipy optimization routine.

    Notes
    -----
    The Levenberg-Marquardt does not support parameter boundaries.
      If lm is True, the routine will find the un-bounded best-fitting
    solution, regardless of pmin and pmax.

    If the model parameters are not bound (i.e., np.all(pmin == -np.inf) and
      np.all(pmax == np.inf)), this code will use the more-efficient
      Levenberg-Marquardt algorithm.

.. py:function:: residuals(fitparams, params, func, data, uncert, indparams, pstep, prior, priorlow, priorup, ifree, ishare, iprior)
.. code-block:: pycon

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
    iprior: 1D bool ndarray
       Indices of the prior parameters in params.

    Returns
    -------
    Array of weighted data-model and prior-params residuals.


MCcubed.mc
__________


.. py:module:: MCcubed.mc

.. py:function:: mcmc(data=None, uncert=None, func=None, indparams=[], params=None, pmin=None, pmax=None, pstep=None, prior=None, priorlow=None, priorup=None, nchains=7, ncpu=None, nsamples=100000.0, walk='snooker', wlike=False, leastsq=False, lm=False, chisqscale=False, grtest=True, grbreak=0.0, grnmin=0.5, burnin=0, thinning=1, fgamma=1.0, fepsilon=0.0, hsize=10, kickoff='normal', plots=False, ioff=False, showbp=True, savefile=None, savemodel=None, resume=False, rms=False, log=None, pnames=None, texnames=None, parname=None, nproc=None, stepsize=None, full_output=None, chireturn=None)
.. code-block:: pycon

    This beautiful piece of code runs a Markov-chain Monte Carlo algorithm.

    Parameters
    ----------
    data: 1D float ndarray or string
        Data to be fit by func.  If string, path to file containing data.
    uncert: 1D float ndarray
        Uncertainties of data.
    func: Callable or string-iterable
        The callable function that models data as:
            model = func(params, *indparams)
        Or an iterable of 3 strings (funcname, modulename, path)
        that specifies the function name, function module, and module path.
        If the module is already in the python-path scope, path can be omitted.
    indparams: tuple or string
        Additional arguments required by func.  If string, path to file
        containing indparams.
    params: 1D/2D float ndarray or string
        Set of initial fitting parameters for func.  If 2D, of shape
        (nparams, nchains), it is assumed that it is one set for each chain.
        If string, path to file containing data.
    pmin: 1D ndarray
       Lower boundaries for the posterior exploration.
    pmax: 1D ndarray
       Upper boundaries for the posterior exploration.
    pstep: 1D ndarray
       Parameter stepping.  If a value is 0, keep the parameter fixed.
       Negative values indicate a shared parameter (See Note 1).
    prior: 1D ndarray
       Parameter prior distribution means (See Note 2).
    priorlow: 1D ndarray
       Lower prior uncertainty values (See Note 2).
    priorup: 1D ndarray
       Upper prior uncertainty values (See Note 2).
    nchains: Scalar
       Number of simultaneous chains to run.
    ncpu: Integer
       Number of processors for the MCMC chains (MC3 defaults to
       one CPU for each chain plus a CPU for the central hub).
    nsamples: Scalar
       Total number of samples.
    walk: String
       Random walk algorithm:
       - 'mrw':  Metropolis random walk.
       - 'demc': Differential Evolution Markov chain.
       - 'snooker': DEMC-z with snooker update.
    wlike: Boolean
       If True, calculate the likelihood in a wavelet-base.  This requires
       three additional parameters (See Note 3).
    leastsq: Boolean
       Perform a least-square minimization before the MCMC run.
    lm: Boolean
       If True use the Levenberg-Marquardt algorithm for the optimization.
       If False, use the Trust Region Reflective algorithm.
    chisqscale: Boolean
       Scale the data uncertainties such that the reduced chi-squared = 1.
    grtest: Boolean
       Run Gelman & Rubin test.
    grbreak: Float
       Gelman-Rubin convergence threshold to stop the MCMC (I'd suggest
       grbreak ~ 1.001--1.005).  Do not break if grbreak=0.0 (default).
    grnmin: Integer or float
       Minimum number of samples required for grbreak to stop the MCMC.
       If grnmin > 1: grnmin sets the minimum required number of samples.
       If 0 < grnmin < 1: grnmin sets the minimum required nsamples fraction.
    burnin: Integer
       Number of burned-in (discarded) number of iterations at the beginning
       of the chains.
    thinning: Integer
       Thinning factor of the chains (use every thinning-th iteration) used
       in the GR test and plots.
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
    ioff: Bool
       If True, set plt.ioff(), i.e., do not display figures on screen.
    showbp: Bool
       If True, show best-fitting values in histogram and pairwise plots.
    savefile: String
       If not None, filename to store allparams and other MCMC results.
    savemodel: String
       If not None, filename to store the values of the evaluated function.
    resume: Boolean
       If True resume a previous run.
    rms: Boolean
       If True, calculate the RMS of the residuals: data - bestmodel.
    log: String or FILE pointer
       Filename or File object to write log.
    pnames: 1D string iterable
       List of parameter names (including fixed and shared parameters)
       to display on output screen and figures.  See also texnames.
       Screen output trims up to the 11th character.
       If not defined, default to texnames.
    texnames: 1D string iterable
       Parameter names for figures, which may use latex syntax.
       If not defined, default to pnames.
    parname: 1D string ndarray
        Deprecated, use pnames instead.
    nproc: Integer
        Deprecated, use ncpu instead.
    stepsize: 1D ndarray
        Deprecated, use pstep instead.
    chireturn:
        Deprecated.
    full_output:  Bool
        Deprecated.

    Returns
    -------
    bestp: 1D ndarray
       Array of the best-fitting parameters (including fixed and shared).
    CRlo:  1D ndarray
       The lower boundary of the marginal 68%-highest posterior density
       (the credible region) for each parameter, with respect to bestp.
    CRhi:  1D ndarray
       The upper boundary of the marginal 68%-highest posterior density
       (the credible region) for each parameter, with respect to bestp.
    stdp: 1D ndarray
       Array of the best-fitting parameter uncertainties, calculated as the
       standard deviation of the marginalized, thinned, burned-in posterior.
    posterior: 2D float ndarray
       An array of shape (Nfreepars, Nsamples) with the thinned MCMC posterior
       distribution of the fitting parameters (excluding fixed and shared).
       If full_output is True, the posterior includes the burnin samples.
    Zchain: 1D integer ndarray
       Index of the chain for each sample in posterior.  M0 samples have chain
       index of -1.
    chiout: 4-elements tuple
       Tuple containing the best-fit chi-square, reduced chi-square, scale
       factor to enforce redchisq=1, and the Bayesian information
       criterion (BIC).

    Notes
    -----
    1.- To set one parameter equal to another, set its pstep to the
        negative index in params (Starting the count from 1); e.g.: to set
        the second parameter equal to the first one, do: pstep[1] = -1.
    2.- If any of the fitting parameters has a prior estimate, e.g.,
          param[i] = p0 +up/-low,
        with up and low the 1sigma uncertainties.  This information can be
        considered in the MCMC run by setting:
        prior[i]    = p0
        priorup[i]  = up
        priorlow[i] = low
        All three: prior, priorup, and priorlow must be set and, furthermore,
        priorup and priorlow must be > 0 to be considered as prior.
    3.- If data, uncert, params, pmin, pmax, pstep, prior, priorlow,
        or priorup are set as filenames, the file must contain one value per
        line.
        For simplicity, the data file can hold both data and uncert arrays.
        In this case, each line contains one value from each array per line,
        separated by an empty-space character.
        Similarly, params can hold: params, pmin, pmax, pstep, priorlow,
        and priorup.  The file can hold as few or as many array as long as
        they are provided in that exact order.
    4.- An indparams file works differently, the file will be interpreted
        as a list of arguments, one in each line.  If there is more than one
        element per line (empty-space separated), it will be interpreted as
        an array.
    5.- FINDME: WAVELET LIKELIHOOD

    Examples
    --------
    >>> # See https://github.com/pcubillos/MCcubed/tree/master/examples

.. py:function:: gelmanrubin(Z, Zchain, burnin)
.. code-block:: pycon

    Gelman & Rubin (1992) convergence test on a MCMC
    chain of parameters.

    Parameters
    ----------
    Z: 2D float ndarray
        A 2D array of shape (nsamples, nparameters) containing
        the parameter MCMC chains.
    Zchain: 1D integer ndarray
        A 1D array of length nsamples indicating the chain for each
        sample.
    burnin: Integer
        Number of iterations to remove.

    Returns
    -------
    GRfactor : 1D float ndarray
        The potential scale reduction factors of the chain for each
        parameter.  If they are much greater than 1, the chain is not
        converging.

    Uncredited developers
    ---------------------
    Chris Campo  (UCF)


MCcubed.plots
_____________


.. py:module:: MCcubed.plots

.. py:function:: trace(posterior, Zchain=None, pnames=None, thinning=1, burnin=0, fignum=100, savefile=None, fmt='.', ms=2.5, fs=11)
.. code-block:: pycon

    Plot parameter trace MCMC sampling.

    Parameters
    ----------
    posterior: 2D float ndarray
       An MCMC posterior sampling with dimension: [nsamples, npars].
    Zchain: 1D integer ndarray
       the chain index for each posterior sample.
    pnames: Iterable (strings)
       Label names for parameters.
    thinning: Integer
       Thinning factor for plotting (plot every thinning-th value).
    burnin: Integer
       Thinned burn-in number of iteration (only used when Zchain is not None).
    fignum: Integer
       The figure number.
    savefile: Boolean
       If not None, name of file to save the plot.
    fmt: String
       The format string for the line and marker.
    ms: Float
       Marker size.
    fs: Float
       Fontsize of texts.

    Returns
    -------
    axes: 1D axes ndarray
       The array of axes containing the marginal posterior distributions.

    Uncredited Developers
    ---------------------
    Kevin Stevenson  (UCF)

.. py:function:: pairwise(posterior, pnames=None, thinning=1, fignum=200, savefile=None, bestp=None, nbins=35, nlevels=20, absolute_dens=False, ranges=None, fs=11, rect=None, margin=0.01)
.. code-block:: pycon

    Plot parameter pairwise posterior distributions.

    Parameters
    ----------
    posterior: 2D ndarray
       An MCMC posterior sampling with dimension: [nsamples, nparameters].
    pnames: Iterable (strings)
       Label names for parameters.
    thinning: Integer
       Thinning factor for plotting (plot every thinning-th value).
    fignum: Integer
       The figure number.
    savefile: Boolean
       If not None, name of file to save the plot.
    bestp: 1D float ndarray
       If not None, plot the best-fitting values for each parameter
       given by bestp.
    nbins: Integer
       The number of grid bins for the 2D histograms.
    nlevels: Integer
       The number of contour color levels.
    ranges: List of 2-element arrays
       List with custom (lower,upper) x-ranges for each parameter.
       Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
    fs: Float
       Fontsize of texts.
    rect: 1D list/ndarray
       If not None, plot the pairwise plots in current figure, within the
       ranges defined by rect (xleft, ybottom, xright, ytop).
    margin: Float
       Margins between panels (when rect is not None).

    Returns
    -------
    axes: 2D axes ndarray
       The grid of axes containing the pairwise posterior distributions.
    cb: axes
       The colorbar axes instance.

    Notes
    -----
    Note that rect delimits the boundaries of the panels. The labels and
    ticklabels will appear right outside rect, so the user needs to leave
    some wiggle room for them.

    Uncredited Developers
    ---------------------
    Kevin Stevenson  (UCF)
    Ryan Hardy       (UCF)

.. py:function:: histogram(posterior, pnames=None, thinning=1, fignum=300, savefile=None, bestp=None, percentile=None, pdf=None, xpdf=None, ranges=None, axes=None, lw=2.0, fs=11)
.. code-block:: pycon

    Plot parameter marginal posterior distributions

    Parameters
    ----------
    posterior: 1D or 2D float ndarray
       An MCMC posterior sampling with dimension [nsamples] or
       [nsamples, nparameters].
    pnames: Iterable (strings)
       Label names for parameters.
    thinning: Integer
       Thinning factor for plotting (plot every thinning-th value).
    fignum: Integer
       The figure number.
    savefile: Boolean
       If not None, name of file to save the plot.
    bestp: 1D float ndarray
       If not None, plot the best-fitting values for each parameter
       given by bestp.
    percentile: Float
       If not None, plot the percentile- highest posterior density region
       of the distribution.  Note that this should actually be the
       fractional part, i.e. set percentile=0.68 for a 68% HPD.
    pdf: 1D float ndarray or list of ndarrays
       A smoothed PDF of the distribution for each parameter.
    xpdf: 1D float ndarray or list of ndarrays
       The X coordinates of the PDFs.
    ranges: List of 2-element arrays
       List with custom (lower,upper) x-ranges for each parameter.
       Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
    axes: List of matplotlib.axes
       If not None, plot histograms in the currently existing axes.
    lw: Float
       Linewidth of the histogram contour.
    fs: Float
       Font size for texts.

    Returns
    -------
    axes: 1D axes ndarray
       The array of axes containing the marginal posterior distributions.

    Uncredited Developers
    ---------------------
    Kevin Stevenson  (UCF)

.. py:function:: RMS(binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1, timepoints=[], ratio=False, fignum=-40, yran=None, xran=None, savefile=None)
.. code-block:: pycon

    Plot the RMS vs binsize curve.

    Parameters
    ----------
    binsz: 1D ndarray
       Array of bin sizes.
    rms: 1D ndarray
       RMS of dataset at given binsz.
    stderr: 1D ndarray
       Gaussian-noise rms Extrapolation
    rmslo: 1D ndarray
       RMS lower uncertainty
    rmshi: 1D ndarray
       RMS upper uncertainty
    cadence: Float
       Time between datapoints in seconds.
    binstep: Integer
       Plot every-binstep point.
    timepoints: List
       Plot a vertical line at each time-points.
    ratio: Boolean
       If True, plot rms/stderr, else, plot both curves.
    fignum: Integer
       Figure number
    yran: 2-elements tuple
       Minimum and Maximum y-axis ranges.
    xran: 2-elements tuple
       Minimum and Maximum x-axis ranges.
    savefile: String
       If not None, name of file to save the plot.

.. py:function:: modelfit(data, uncert, indparams, model, nbins=75, fignum=-50, savefile=None, fmt='.')
.. code-block:: pycon

    Plot the binned dataset with given uncertainties and model curves
    as a function of indparams.
    In a lower panel, plot the residuals bewteen the data and model.

    Parameters
    ----------
    data:  1D float ndarray
      Input data set.
    uncert:  1D float ndarray
      One-sigma uncertainties of the data points.
    indparams:  1D float ndarray
      Independent variable (X axis) of the data points.
    model:  1D float ndarray
      Model of data.
    nbins:  Integer
      Number of bins in the output plot.
    fignum:  Integer
      The figure number.
    savefile:  Boolean
      If not None, name of file to save the plot.
    fmt:  String
      Format of the plotted markers.

.. py:function:: subplotter(rect, margin, ipan, nx, ny=None, ymargin=None)
.. code-block:: pycon

    Create an axis instance for one panel (with index ipan) of a grid
    of npanels, where the grid located inside rect (xleft, ybottom,
    xright, ytop).

    Parameters
    ----------
    rect: 1D List/ndarray
       Rectangle with xlo, ylo, xhi, yhi positions of the grid boundaries.
    margin: Float
       Width of margin between panels.
    ipan: Integer
       Index of panel to create (as in plt.subplots).
    nx: Integer
       Number of panels along the x axis.
    ny: Integer
       Number of panels along the y axis. If None, assume ny=nx.
    ymargin: Float
       Width of margin between panels along y axes (if None, adopt margin).

    Returns
    -------
    axes: axes instance
       A matplotlib axes instance at the specified position.


MCcubed.utils
_____________


.. py:module:: MCcubed.utils

.. py:data:: ROOT
.. code-block:: pycon

  '/home/pcubillos/Dropbox/IWF/projects/2014_mc3/multiproc/MCcubed/'

.. py:function:: parray(string)
.. code-block:: pycon

    Convert a string containin a list of white-space-separated (and/or
    newline-separated) values into a numpy array

.. py:function:: saveascii(data, filename, precision=8)
.. code-block:: pycon

    Write (numeric) data to ASCII file.

    Parameters
    ----------
    data:  1D/2D numeric iterable (ndarray, list, tuple, or combination)
        Data to be stored in file.
    filename:  String
        File where to store the arrlist.
    precision: Integer
        Maximum number of significant digits of values.

    Example
    -------
    >>> import numpy as np
    >>> import MCcubed.utils as mu

    >>> a = np.arange(4) * np.pi
    >>> b = np.arange(4)
    >>> c = np.logspace(0, 12, 4)

    >>> outfile = 'delete.me'
    >>> mu.saveascii([a,b,c], outfile)

    >>> # This will produce this file:
    >>> with open(outfile) as f:
    >>>   print(f.read())
            0         0         1
    3.1415927         1     10000
    6.2831853         2     1e+08
     9.424778         3     1e+12

.. py:function:: loadascii(filename)
.. code-block:: pycon

    Extract data from file and store in a 2D ndarray (or list of arrays
    if not square).  Blank or comment lines are ignored.

    Parameters
    ----------
    filename: String
        Name of file containing the data to read.

    Returns
    -------
    array: 2D ndarray or list
        See parameters description.

.. py:function:: savebin(data, filename)
.. code-block:: pycon

    Write data variables into a numpy npz file.

    Parameters
    ----------
    data:  List of data objects
        Data to be stored in file.  Each array must have the same length.
    filename:  String
        File where to store the arrlist.

    Note
    ----
    This wrapper around np.savez() preserves the data type of list and
    tuple variables when the file is open with loadbin().

    Example
    -------
    >>> import MCcubed.utils as mu
    >>> import numpy as np
    >>> # Save list of data variables to file:
    >>> datafile = 'datafile.npz'
    >>> indata = [np.arange(4), 'one', np.ones((2,2)), True, [42], (42, 42)]
    >>> mu.savebin(indata, datafile)
    >>> # Now load the file:
    >>> outdata = mu.loadbin(datafile)
    >>> for data in outdata:
    >>>     print(repr(data))
    array([0, 1, 2, 3])
    'one'
    array([[ 1.,  1.],
           [ 1.,  1.]])
    True
    [42]
    (42, 42)

.. py:function:: loadbin(filename)
.. code-block:: pycon

    Read a binary npz array, casting list and tuple variables into
    their original data types.

    Parameters
    ----------
    filename: String
       Path to file containing the data to be read.

    Return
    ------
    data:  List
       List of objects stored in the file.

    Example
    -------
    See example in savebin().

.. py:function:: isfile(input, iname, log, dtype, unpack=True, not_none=False)
.. code-block:: pycon

    Check if an input is a file name; if it is, read it.
    Genereate error messages if it is the case.

    Parameters
    ----------
    input: Iterable or String
        The input variable.
    iname: String
        Input-variable name.
    log: File pointer
         If not None, print message to the given file pointer.
    dtype: String
        File data type, choose between 'bin' or 'ascii'.
    unpack: Bool
        If True, return the first element of a read file.
    not_none: Bool
        If True, throw an error if input is None.

.. py:function:: binarray(...)
.. code-block:: pycon

    Compute the weighted-mean binned values and uncertainties of an array.
                                                                
    Parameters                                                      
    ----------                                                      
    data: 1D ndarray                                                
        A time-series dataset.                                      
    uncert: 1D ndarray                                              
        Uncertainties of data.                                      
    indp: 1D ndarray                                                
        Independent variable.                                       
    binsize: Integer                                                
        Number of data points per bin.                              
                                                                
    Returns                                                         
    -------                                                         
    bindata: 1D ndarray                                             
        Mean-weighted binned data (using 1/unc**2 as weights).      
    binunc: 1D ndarray                                              
        Standard deviation of the binned data points.               
    binindp: 1D ndarray                                             
        Mean-averaged binned indp.                                  
                                                                
    Examples                                                        
    --------                                                        
    import MCcubed.utils as mu                                      
    ndata = 12                                                      
    data   = np.arange(ndata, dtype=np.double)                      
    uncert = np.ones(ndata)                                         
    indp   = np.arange(ndata, dtype=np.double)                      
    bindata, binunc, binx = mu.binarray(data, uncert, indp, binsize)
                                                                
    Uncredited Developers                                           
    ---------------------                                           
    Kevin Stevenson (UCF)                                           
    Matt Hardin (UCF)

.. py:function:: weightedbin(...)
.. code-block:: pycon

    Calculate the weighted mean (for known bin standard deviation)   
                                                                
    Parameters                                                      
    ----------                                                      
    data: 1D ndarray                                                
      A time-series dataset.                                        
    binsize: Integer                                                
      Number of data points per bin.                                
    uncert: 1D ndarray                                              
      Uncertainties of data.                                        
    var: 1D ndarray                                                 
      Variance of the bins (=1/sum(1/uncert**2.0) for any given bin).
                                                                
    Notes                                                           
    -----                                                           
    If uncert and std are not provided, use flat weights.           
                                                                
    See Equation (4.31) of Data Reduction and Error Analysis        
    for the Physical Sciences (Bevington, Robinson).                
                                                                
    Returns                                                         
    -------                                                         
    bindat: 1D ndarray                                              
       Mean-weighted binned data (using 1/uncert**2 as weights).

.. py:function:: credregion(posterior=None, percentile=0.6827, pdf=None, xpdf=None)
.. code-block:: pycon

    Compute a smoothed posterior density distribution and the minimum
    density for a given percentile of the highest posterior density.

    These outputs can be used to easily compute the HPD credible regions.

    Parameters
    ----------
    posterior: 1D float ndarray
        A posterior distribution.
    percentile: Float
        The percentile (actually the fraction) of the credible region.
        A value in the range: (0, 1).
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.

    Returns
    -------
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    HPDmin: Float
        The minimum density in the percentile-HPD region.

    Example
    -------
    >>> import numpy as np
    >>> import MCcubed.utils as mu
    >>> # Test for a Normal distribution:
    >>> npoints = 100000
    >>> posterior = np.random.normal(0, 1.0, npoints)
    >>> pdf, xpdf, HPDmin = mu.credregion(posterior)
    >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

    >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
    >>> pdf, xpdf, HPDmin = mu.credregion(pdf=pdf, xpdf=xpdf, percentile=0.9545)
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

.. py:function:: burn(Zdict=None, burnin=None, Z=None, Zchain=None, sort=True)
.. code-block:: pycon

    Return a posterior distribution removing the burnin initial iterations
    of each chain from the input distribution.

    Parameters
    ----------
    Zdict: dict
        A dictionary (as in MC3's output) containing a posterior distribution
        (Z) and number of iterations to burn (burnin).
    burnin: Integer
        Number of iterations to remove from the start of each chain.
        If specified, it overrides value from Zdict.
    Z: 2D float ndarray
        Posterior distribution (of shape [nsamples,npars]) to consider
        if Zdict is None.
    Zchain: 1D integer ndarray
        Chain indices for the samples in Z (used only of Zdict is None).
    sort: Bool
        If True, sort the outputs by chain index.

    Returns
    -------
    posterior: 2D float ndarray
        Burned posterior distribution.
    Zchain: 1D integer ndarray
        Burned Zchain array.
    Zmask: 1D integer ndarray
        Indices that transform Z into posterior.

    Examples
    --------
    >>> import MCcubed.utils as mu
    >>> import numpy as np
    >>> # Mock a posterior-distribution output:
    >>> Z = np.expand_dims([0., 1, 10, 20, 30, 11, 31, 21, 12, 22, 32], axis=1)
    >>> Zchain = np.array([-1, -1, 0, 1, 2, 0, 2, 1, 0, 1, 2])
    >>> Zdict = {'Z':Z, 'Zchain':Zchain, 'burnin':1}
    >>> # Simply apply burn() into the dict:
    >>> posterior, zchain, zmask = mu.burn(Zdict)
    >>> print(posterior[:,0])
    [11. 12. 21. 22. 31. 32.]
    >>> print(zchain)
    [0 0 1 1 2 2]
    >>> print(zmask)
    [ 5  8  7  9  6 10]
    >>> # Samples were sorted by chain index, but one can prevent with:
    >>> posterior, zchain, zmask = mu.burn(Zdict, sort=False)
    >>> print(posterior[:,0])
    [11. 31. 21. 12. 22. 32.]
    >>> # One can also override the burn-in samples:
    >>> posterior, zchain, zmask = mu.burn(Zdict, burnin=0)
    >>> print(posterior[:,0])
    [10. 11. 12. 20. 21. 22. 30. 31. 32.]
    >>> # Or apply directly to arrays:
    >>> posterior, zchain, zmask = mu.burn(Z=Z, Zchain=Zchain, burnin=1)
    >>> print(posterior[:,0])
    [11. 12. 21. 22. 31. 32.]

.. py:function:: default_parnames(npars)
.. code-block:: pycon

    Create an array of parameter names with sequential indices.

    Parameters
    ----------
    npars: Integer
        Number of parameters.

    Results
    -------
    1D string ndarray of parameter names.

.. py:class:: Log(logname, verb=2, append=False, width=70)

.. code-block:: pycon

    Dual file/stdout logging class with conditional printing.

  .. code-block:: pycon

    Parameters
    ----------
    logname: String
        Name of FILE pointer where to store log entries. Set to None to
        print only to stdout.
    verb: Integer
        Conditional threshold to print messages.  There are five levels
        of increasing verbosity:
        verb <  0: only print error() calls.
        verb >= 0: print warning() calls.
        verb >= 1: print head() calls.
        verb >= 2: print msg() calls.
        verb >= 3: print debug() calls.
    append: Bool
        If True, append logged text to existing file.
        If False, write logs to new file.
    width: Integer
        Maximum length of each line of text (longer texts will be break
        down into multiple lines).


MCcubed.rednoise
________________


.. py:module:: MCcubed.rednoise

.. py:function:: binrms(...)
.. code-block:: pycon

    Compute the binned root-mean-square and extrapolated           
    Gaussian-noise rms for a dataset.                               
                                                                
    Parameters                                                      
    ----------                                                      
    data: 1D ndarray                                                
      A time-series dataset.                                        
    maxbins: Scalar                                                 
      Maximum bin size to calculate.                                
    binstep: Integer                                                
      Stepsize of binning indexing.                                 
                                                                
    Returns                                                         
    -------                                                         
    rms: 1D ndarray                                                 
       RMS of binned data.                                          
    rmslo: 1D ndarray                                               
       RMS lower uncertainties.                                     
    rmshi: 1D ndarray                                               
       RMS upper uncertainties.                                     
    stderr: 1D ndarray                                              
       Extrapolated RMS for Gaussian noise.                         
    binsz: 1D ndarray                                               
       Bin sizes.                                                   
                                                                
    Notes                                                           
    -----                                                           
    This function uses an asymptotic approximation to obtain the    
    rms uncertainties (rms_error = rms/sqrt(2M)) when the number of 
    bins is M > 35.                                                 
    At smaller M, the errors become increasingly asymmetric. In this
    case the errors are numerically calculated from the posterior   
    PDF of the rms (an inverse-gamma distribution).                 
    See Cubillos et al. (2016).                                     
                                                                
    Uncredited developers                                           
    ---------------------                                           
    Kevin Stevenson (UCF)                                           
    Matt Hardin (UCF)

.. py:function:: prayer(configfile=None, nprays=0, savefile=None)
.. code-block:: pycon

    Implement a prayer-bead method to estimate parameter uncertainties.

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

