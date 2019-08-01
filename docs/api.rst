API
===


mc3
___


.. py:module:: mc3

.. py:function:: mcmc(data=None, uncert=None, func=None, params=None, indparams=[], pmin=None, pmax=None, pstep=None, prior=None, priorlow=None, priorup=None, nchains=7, ncpu=None, nsamples=None, sampler=None, wlike=False, leastsq=None, chisqscale=False, grtest=True, grbreak=0.0, grnmin=0.5, burnin=0, thinning=1, fgamma=1.0, fepsilon=0.0, hsize=10, kickoff='normal', plots=False, ioff=False, showbp=True, savefile=None, resume=False, rms=False, log=None, pnames=None, texnames=None, parname=None, nproc=None, stepsize=None, full_output=None, chireturn=None, lm=None, walk=None)
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
    params: 1D/2D float ndarray or string
        Set of initial fitting parameters for func.  If 2D, of shape
        (nparams, nchains), it is assumed that it is one set for each chain.
        If string, path to file containing data.
    indparams: tuple or string
        Additional arguments required by func.  If string, path to file
        containing indparams.
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
    sampler: String
        Sampler algorithm:
        - 'mrw':  Metropolis random walk.
        - 'demc': Differential Evolution Markov chain.
        - 'snooker': DEMC-z with snooker update.
    wlike: Bool
        If True, calculate the likelihood in a wavelet-base.  This requires
        three additional parameters (See Note 3).
    leastsq: String
        If not None, perform a least-square optimization before the MCMC run.
        Select from:
            'lm':  Levenberg-Marquardt (most efficient, but does not obey bounds)
            'trf': Trust Region Reflective
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
    resume: Boolean
        If True resume a previous run.
    rms: Boolean
        If True, calculate the RMS of the residuals: data - best_model.
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
    lm: Bool
        Deprecated, see leastsq.
    walk: String
        Deprecated, use sampler instead.

    Returns
    -------
    mc3_output: Dict
        A Dictionary containing the MCMC posterior distribution and related
        stats, including:
        - Z: thinned posterior distribution of shape [nsamples, nfree].
        - Zchain: chain indices for each sample in Z.
        - Zchisq: chi^2 value for each sample in Z.
        - Zmask: indices that turn Z into the desired posterior (remove burn-in).
        - burnin: number of burned-in samples per chain.
        - meanp: mean of the marginal posteriors.
        - stdp: standard deviation of the marginal posteriors.
        - CRlo: lower boundary of the marginal 68%-highest posterior
              density (the credible region).
        - CRhi: upper boundary of the marginal 68%-HPD.
        - bestp: model parameters for the lowest-chi^2 sample.
        - best_model: model evaluated at bestp.
        - best_chisq: lowest-chi^2 in the sample.
        - red_chisq: reduced chi-squared: chi^2/(Ndata}-Nfree) for the
              best-fitting sample.
        - BIC: Bayesian Information Criterion: chi^2-Nfree log(Ndata)
              for the best-fitting sample.
        - chisq_factor: Uncertainties scale factor to enforce chi^2_red = 1.
        - stddev_residuals: standard deviation of the residuals.
        - acceptance_rate: sample's acceptance rate.

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
    >>> # See https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html

.. py:function:: nested_sampling(data=None, uncert=None, func=None, params=None, indparams=[], pmin=None, pmax=None, pstep=None, prior=None, priorlow=None, priorup=None, ncpu=1, nsamples=None, sampler=None, leastsq=None, chisqscale=False, thinning=1, plots=False, ioff=False, showbp=True, savefile=None, resume=False, rms=False, log=None, pnames=None, texnames=None)
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
      sampler: String
          Sampler algorithm:
          - 'mrw':  Metropolis random walk.
          - 'demc': Differential Evolution Markov chain.
          - 'snooker': DEMC-z with snooker update.
      wlike: Bool
          If True, calculate the likelihood in a wavelet-base.  This requires
          three additional parameters (See Note 3).
      leastsq: String
          If not None, perform a least-square optimization before the MCMC run.
          Select from:
              'lm':  Levenberg-Marquardt (most efficient, but does not obey bounds)
              'trf': Trust Region Reflective
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
      resume: Boolean
          If True resume a previous run.
      rms: Boolean
          If True, calculate the RMS of the residuals: data - best_model.
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

      Returns
      -------
      mc3_output: Dict
          A Dictionary containing the MCMC posterior distribution and related
          stats, including:
          - Z: thinned posterior distribution of shape [nsamples, nfree].
          - Zchain: chain indices for each sample in Z.
          - Zchisq: chi^2 value for each sample in Z.
          - Zmask: indices that turn Z into the desired posterior.
          - burnin: number of burned-in samples per chain.
          - CRlo: lower boundary of the marginal 68%-highest posterior
                density (the credible region).
          - CRhi: upper boundary of the marginal 68%-HPD.
          - stdp: standard deviation of the marginal posteriors.
          - meanp: mean of the marginal posteriors.
          - bestp: model parameters for the lowest-chi^2 sample.
          - best_chisq: lowest-chi^2 in the sample.
          - best_model: model evaluated at bestp.
          - red_chisq: reduced chi-squared: chi^2/(Ndata}-Nfree) for the
                best-fitting sample.
          - BIC: Bayesian Information Criterion: chi^2-Nfree log(Ndata)
                for the best-fitting sample.
          - chisq_factor: Uncertainties scale factor to enforce chi^2_red = 1.
          - stddev_residuals: standard deviation of the residuals.
          - acceptance_rate: sample's acceptance rate.

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
      >>> # See https://mc3.readthedocs.io/en/latest/ns_tutorial.html

    import numpy as np
    import mc3

    def quad(p, x):
        return p[0] + p[1]*x + p[2]*x**2.0

    # Create a noisy synthetic dataset:
    x = np.linspace(0, 10, 100)
    p_true = [3, -2.4, 0.5]
    y = quad(p_true, x)
    uncert = np.sqrt(np.abs(y))
    error = np.random.normal(0, uncert)
    data = y + error

    # Initial guess for fitting parameters:
    params = np.array([3.0, -2.0, 0.1])
    pstep  = np.array([0.0, 0.03, 0.05])
    pmin   = np.array([ 0.0, -5.0, -1.0])
    pmax   = np.array([20.0,  5.0,  1.0])

    indparams = [x]
    func = quad
    ncpu = 4

    mc3_results = mc3.nested_sampling(data, uncert, func=quad, params=params,
        indparams=[x], pstep=pstep, ncpu=ncpu, pmin=pmin, pmax=pmax, leastsq='lm')

    mc3_mcmc = mc3.mcmc(data, uncert, func=quad, params=params, indparams=[x],
        pstep=pstep, ncpu=ncpu, pmin=pmin, pmax=pmax, leastsq='lm')
  

.. py:function:: fit(data, uncert, func, params, indparams=[], pstep=None, pmin=None, pmax=None, prior=None, priorlow=None, priorup=None, leastsq='lm')
.. code-block:: pycon

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


mc3.plots
_________


.. py:module:: mc3.plots

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
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.

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
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.

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
    axes: 2D ndarray of matplotlib.axes.Axes
        Array of axes containing the marginal posterior distributions.
    cb: matplotlib.axes.Axes
        The colorbar axes.

    Notes
    -----
    rect delimits the boundaries of the panels. The labels and
    ticklabels will appear outside rect, so the user needs to leave
    some wiggle room for them.

.. py:function:: rms(binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1, timepoints=[], ratio=False, fignum=410, yran=None, xran=None, savefile=None)
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

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes instance containing the marginal posterior distributions.

.. py:function:: modelfit(data, uncert, indparams, model, nbins=75, fignum=411, savefile=None, fmt='.')
.. code-block:: pycon

    Plot the binned dataset with given uncertainties and model curves
    as a function of indparams.
    In a lower panel, plot the residuals bewteen the data and model.

    Parameters
    ----------
    data: 1D float ndarray
        Input data set.
    uncert: 1D float ndarray
        One-sigma uncertainties of the data points.
    indparams: 1D float ndarray
        Independent variable (X axis) of the data points.
    model: 1D float ndarray
        Model of data.
    nbins: Integer
        Number of bins in the output plot.
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    fmt: String
        Format of the plotted markers.

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes instance containing the marginal posterior distributions.

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
    axes: Matplotlib.axes.Axes
        An Axes instance at the specified position.


mc3.utils
_________


.. py:module:: mc3.utils

.. py:data:: ROOT
.. code-block:: pycon

  '/Users/pato/Dropbox/IWF/projects/2014_mc3/multiproc/MCcubed/'

.. py:function:: ignore_system_exit(func)
.. code-block:: pycon

    Decorator to ignore SystemExit exceptions.

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
    >>> import mc3.utils as mu

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
    >>> import mc3.utils as mu
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
    >>> import mc3.utils as mu
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

.. py:function:: credregion(posterior=None, percentile=0.6827, pdf=None, xpdf=None)
.. code-block:: pycon

    Compute the highest-posterior-density credible region for a
    posterior distribution.

    This function has been deprecated.  Use mc3.stats.cred_region()
    instead.

.. py:class:: Log(logname=None, verb=2, append=False, width=70)

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


mc3.stats
_________


.. py:module:: mc3.stats

.. py:function:: gelman_rubin(Z, Zchain, burnin)
.. code-block:: pycon

    Gelman--Rubin convergence test on a MCMC chain of parameters
    (Gelman & Rubin, 1992).

    Parameters
    ----------
    Z: 2D float ndarray
        A 2D array of shape (nsamples, npars) containing
        the parameter MCMC chains.
    Zchain: 1D integer ndarray
        A 1D array of length nsamples indicating the chain for each
        sample.
    burnin: Integer
        Number of iterations to remove.

    Returns
    -------
    GRfactor: 1D float ndarray
        The potential scale reduction factors of the chain for each
        parameter.  If they are much greater than 1, the chain is not
        converging.

.. py:function:: bin_array(data, binsize, uncert=None)
.. code-block:: pycon

    Compute the binned weighted mean and standard deviation of an array
    using 1/uncert**2 as weights.
    Eq. (4.31) of Data Reduction and Error Analysis for the Physical
    Sciences by Bevington & Robinson).

    Parameters
    ----------
    data: 1D ndarray
        A time-series dataset.
    binsize: Integer
        Number of data points per bin.
    uncert: 1D ndarray
        Uncertainties of data (if None, assume that all data points have
        same uncertainty).

    Returns
    -------
    bindata: 1D ndarray
        Mean-weighted binned data.
    binunc: 1D ndarray
        Standard deviation of the binned data points (returned only if
        uncert is not None).

    Notes
    -----
    If the last bin does not contain binsize elements, it will be
    trnucated from the output.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ndata = 12
    >>> data   = np.array([0,1,2, 3,3,3, 3,3,4])
    >>> uncert = np.array([3,1,1, 1,2,3, 2,2,4])
    >>> binsize = 3
    >>> # Binning, no weights:
    >>> bindata = ms.bin_array(data, binsize)
    >>> print(bindata)
    [1.         3.         3.33333333]
    >>> # Binning using uncertainties as weights:
    >>> bindata, binstd = ms.bin_array(data, binsize, uncert)
    >>> print(bindata)
    [1.42105263 3.         3.11111111]
    >>> print(binstd)
    [0.6882472  0.85714286 1.33333333]

.. py:function:: residuals(model, data, uncert, params=None, priors=None, priorlow=None, priorup=None)
.. code-block:: pycon

    Calculate the residuals between a dataset and a model

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    errors: 1D ndarray
        Data uncertainties.
    params: 1D float ndarray
        Model parameters.
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Prior lower uncertainty.
    priorup: 1D ndarray
        Prior upper uncertainty.

    Returns
    -------
    residuals: 1D ndarray
        Residuals array.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data   = np.array([1.1, 1.2, 0.9, 1.0])
    >>> model  = np.array([1.0, 1.0, 1.0, 1.0])
    >>> uncert = np.array([0.1, 0.1, 0.1, 0.1])
    >>> res = ms.residuals(model, data, uncert)
    print(res)
    [-1. -2.  1.  0.]
    >>> # Now, say this is a two-parameter model, with a uniform and
    >>> # a Gaussian prior, respectively:
    >>> params = np.array([2.5, 5.5])
    >>> priors = np.array([2.0, 5.0])
    >>> plow   = np.array([0.0, 1.0])
    >>> pup    = np.array([0.0, 1.0])
    >>> res = ms.residuals(model, data, uncert, params, priors, plow, pup)
    >>> print(res)
    [-1.  -2.   1.   0.   0.5]

.. py:function:: chisq(model, data, uncert, params=None, priors=None, priorlow=None, priorup=None)
.. code-block:: pycon

    Calculate chi-squared of a model fit to a data set:
        chisq = sum{data points} ((data[i] -model[i])/error[i])**2.0

    If params, priors, priorlow, and priorup are not None, calculate:
        chisq = sum{data points} ((data[i] -model[i])/error[i])**2.0
              + sum{priors} ((params[j]-prior[j])/prioruncert[j])**2.0
    Which is not chi-squared, but is the quantity to optimize when a
    parameter has a Gaussian prior (equivalent to maximize the Bayesian
    posterior probability).

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    uncert: 1D ndarray
        Data uncertainties.
    params: 1D float ndarray
        Model parameters.
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        The chi-squared value.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data   = np.array([1.1, 1.2, 0.9, 1.0])
    >>> model  = np.array([1.0, 1.0, 1.0, 1.0])
    >>> uncert = np.array([0.1, 0.1, 0.1, 0.1])
    >>> chisq  = ms.chisq(model, data, uncert)
    print(chisq)
    6.0
    >>> # Now, say this is a two-parameter model, with a uniform and
    >>> # a Gaussian prior, respectively:
    >>> params = np.array([2.5, 5.5])
    >>> priors = np.array([2.0, 5.0])
    >>> plow   = np.array([0.0, 1.0])
    >>> pup    = np.array([0.0, 1.0])
    >>> chisq = ms.chisq(model, data, uncert, params, priors, plow, pup)
    >>> print(chisq)
    6.25

.. py:function:: dwt_chisq(model, data, params, priors=None, priorlow=None, priorup=None)
.. code-block:: pycon

    Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared)
    based on Carter & Winn (2009), ApJ 704, 51.

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    params: 1D float ndarray
        Model parameters (including the tree noise parameters: gamma,
        sigma_r, sigma_w; which must be the last three elements in params).
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        Wavelet-based (pseudo) chi-squared.

    Notes
    -----
    - If the residuals array size is not of the form 2**N, the routine
    zero-padds the array until this condition is satisfied.
    - The current code only supports gamma=1.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np

    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
    >>> model = np.ones(8)
    >>> params = np.array([1.0, 0.1, 0.1])
    >>> chisq = ms.chisq(model, data, params)
    >>> print(chisq)
    1693.22308882

.. py:function:: cred_region(posterior=None, quantile=0.6827, pdf=None, xpdf=None, percentile=None)
.. code-block:: pycon

    Compute the highest-posterior-density credible region for a
    posterior distribution.

    Parameters
    ----------
    posterior: 1D float ndarray
        A posterior distribution.
    quantile: Float
        The HPD quantile considered for the credible region.
        A value in the range: (0, 1).
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    percentile: Float
        Deprecated. Use quantile instead.

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
    >>> import mc3.stats as ms
    >>> # Test for a Normal distribution:
    >>> npoints = 100000
    >>> posterior = np.random.normal(0, 1.0, npoints)
    >>> pdf, xpdf, HPDmin = ms.cred_region(posterior)
    >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

    >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
    >>> pdf, xpdf, HPDmin = ms.cred_region(pdf=pdf, xpdf=xpdf, quantile=0.9545)
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

.. py:function:: ppf_uniform(pmin, pmax)
.. code-block:: pycon

    Percent-point function (PPF) for a uniform function between
    pmin and pmax.  Also known as inverse CDF or quantile function.

    Parameters
    ----------
    pmin: Float
        Lower boundary of the uniform function.
    pmax: Float
        Upper boundary of the uniform function.

    Returns
    -------
    ppf: Callable
        The uniform's PPF.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ppf_u = ms.ppf_uniform(-10.0, 10.0)
    >>> # The domain of the output function is [0,1]:
    >>> ppf_u(0.0), ppf_u(0.5), ppf_u(1.0)
    (-10.0, 0.0, 10.0)
    >>> # Also works for np.array inputs:
    >>> print(ppf_u(np.array([0.0, 0.5, 1.0])))
    array([-10.,   0.,  10.])

.. py:function:: ppf_gaussian(loc, lo, up)
.. code-block:: pycon

    Percent-point function (PPF) for a two-sided Gaussian function
    Also known as inverse CDF or quantile function.

    Parameters
    ----------
    loc: Float
        Center of the Gaussian function.
    lo: Float
        Left-sided standard deviation (for values x < loc).
    up: Float
        Right-sided standard deviation (for values x > loc).

    Returns
    -------
    ppf: Callable
        The Gaussian's PPF.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> ppf_g = ms.ppf_gaussian(0.0, 1.0, 1.0)
    >>> # The domain of the output function is [0,1]:
    >>> ppf_g(1e-10), ppf_g(0.5), ppf_g(1.0-1e-10)
    (-6.361340902404056, 0.0, 6.361340889697422)
    >>> # Also works for np.array inputs:
    >>> print(ppf_g(np.array([1e-10, 0.5, 1-1e-10])))
    [-6.3613409   0.          6.36134089]

.. py:function:: dwt_daub4(array, inverse=False)
.. code-block:: pycon

    1D discrete wavelet transform using the Daubechies 4-parameter wavelet

    Parameters
    ----------
    array: 1D ndarray
        Data array to which to apply the DWT.
    inverse: bool
        If False, calculate the DWT,
        If True, calculate the inverse DWT.

    Notes
    -----
    The input vector must have length 2**M with M an integer, otherwise
    the output will zero-padded to the next size of the form 2**M.

    Examples
    --------
    >>> import numpy as np
    >>> improt matplotlib.pyplot as plt
    >>> import mc3.stats as ms

    >>> # Calculate the inverse DWT for a unit vector:
    >>> nx = 1024
    >>> e4 = np.zeros(nx)
    >>> e4[4] = 1.0
    >>> ie4 = ms.dwt_daub4(e4, True)
    >>> # Plot the inverse DWT:
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(np.arange(nx), ie4)

.. py:function:: time_avg(data, maxbins=None, binstep=1)
.. code-block:: pycon

    Compute the binned root-mean-square and extrapolated
    Gaussian-noise RMS for a dataset.

    Parameters
    ----------
    data: 1D float ndarray
        A time-series dataset.
    maxbins: Integer
        Maximum bin size to calculate, default: len(data)/2.
    binstep: Integer
        Stepsize of binning indexing.

    Returns
    -------
    rms: 1D float ndarray
        RMS of binned data.
    rmslo: 1D float ndarray
        RMS lower uncertainties.
    rmshi: 1D float ndarray
        RMS upper uncertainties.
    stderr: 1D float ndarray
        Extrapolated RMS for Gaussian noise.
    binsz: 1D float ndarray
        Bin sizes.

    Notes
    -----
    This function uses an asymptotic approximation to obtain the
    rms uncertainties (rms_error = rms/sqrt(2M)) when the number of
    bins is M > 35.
    At smaller M, the errors become increasingly asymmetric. In this
    case the errors are numerically calculated from the posterior
    PDF of the rms (an inverse-gamma distribution).
    See Cubillos et al. (2017), AJ, 153, 3.


mc3.rednoise
____________


.. py:module:: mc3.rednoise

.. py:function:: binrms(data, maxbins=None, binstep=1)
.. code-block:: pycon

    Compute the binned root-mean-square and extrapolated
    Gaussian-noise RMS for a dataset.

    This function has been deprecated.  Use mc3.stats.time_avg()
    instead.

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

