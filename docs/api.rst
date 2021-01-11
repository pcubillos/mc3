.. _api:

API
===


mc3
___


.. py:module:: mc3

.. py:function:: sample(data=None, uncert=None, func=None, params=None, indparams=[], pmin=None, pmax=None, pstep=None, prior=None, priorlow=None, priorup=None, sampler=None, ncpu=None, leastsq=None, chisqscale=False, nchains=7, nsamples=None, burnin=0, thinning=1, grtest=True, grbreak=0.0, grnmin=0.5, wlike=False, fgamma=1.0, fepsilon=0.0, hsize=10, kickoff='normal', plots=False, ioff=False, showbp=True, savefile=None, resume=False, rms=False, log=None, pnames=None, texnames=None, parname=None, nproc=None, stepsize=None, full_output=None, chireturn=None, lm=None, walk=None, **kwargs)
.. code-block:: pycon

    This beautiful piece of code executes an MCMC or NS posterior sampling.

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
    params: 1D float ndarray or string
        Set of initial fitting parameters for func.
        If string, path to file containing data.
    indparams: tuple or string
        Additional arguments required by func.  If string, path to file
        containing indparams.
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
        - 'dynesty': DynamicNestedSampler() sampler from dynesty.
    ncpu: Integer
        Number of processors for the MCMC chains (MC3 defaults to
        one CPU for each chain plus a CPU for the central hub).
    leastsq: String
        If not None, perform a least-square optimization before the MCMC run.
        Select from:
            'lm':  Levenberg-Marquardt (most efficient, but does not obey bounds)
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
        grbreak ~ 1.001--1.005).  Do not break if grbreak=0.0 (default).
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

    Deprecated Parameters
    ---------------------
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
        - posterior: thinned posterior distribution of shape [nsamples, nfree],
              including the burn-in phase.
        - zchain: chain indices for the posterior samples.
        - zmask: posterior mask to remove the burn-in.
        - chisq: chi^2 values for the posterior samples.
        - log_post: -2*log(posterior) for the posterior samples (see Notes).
        - burnin: number of burned-in samples per chain.
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
        log_post = -2*log(posterior)
                 = -2*log(likelihood) - 2*log(prior)
                 = chi-squared + log_prior
                 = sum_i ((data[i] - model[i])/uncert[i])**2 + log_prior

    with log_prior defined as the negative-log of the prior
    (plus a constant, neglected since it does not affect the optimization):
    For a uniform prior:   log_prior = 0.0
    For a Gaussian prior:  log_prior = ((params - prior)/prior_uncert)**2

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
    >>> mcmc_output = mc3.sample(data, uncert, func, params, indparams=indparams,
    >>>     sampler='snooker', pstep=pstep, ncpu=ncpu, pmin=pmin, pmax=pmax,
    >>>     prior=prior, priorlow=priorlow, priorup=priorup,
    >>>     leastsq='lm', nsamples=1e5, burnin=1000, plots=True)

    >>> # Nested sampling:
    >>> ns_output = mc3.sample(data, uncert, func, params, indparams=indparams,
    >>>     sampler='dynesty', pstep=pstep, ncpu=ncpu, pmin=pmin, pmax=pmax,
    >>>     prior=prior, priorlow=priorlow, priorup=priorup,
    >>>     leastsq='lm', plots=True)

    >>> # See more examples and details at:
    >>> # https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html
    >>> # https://mc3.readthedocs.io/en/latest/ns_tutorial.html

.. py:function:: fit(data, uncert, func, params, indparams=[], pstep=None, pmin=None, pmax=None, prior=None, priorlow=None, priorup=None, leastsq='lm')
.. code-block:: pycon

    Find the best-fitting params values to the dataset by performing a
    Maximum-A-Posteriori optimization.

    This is achieved by minimizing the negative-log posterior:
    log_post = -2*log(posterior)
             = -2*log(likelihood) - 2*log(prior)
             = chi-squared + log_prior
             = sum_i ((data[i] - model[i])/uncert[i])**2 + log_prior

    where we define log_prior as the negative-log of the prior
    (plus a constant, neglected since it does not affect the optimization):
      For a uniform prior:   log_prior = 0.0
      For a Gaussian prior:  log_prior = sum ((params - prior)/prior_uncert)**2

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
        - best_log_post: optimal negative-log of the posterior (as defined above).
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
    >>> print(output['bestp'], output['best_chisq'], output['best_log_post'], sep='\n')
    [ 4.57471072 -2.28357843  0.48341911]
    92.79923183159411
    92.79923183159411

    >>> # Fit with priors (Gaussian, uniform, uniform):
    >>> prior    = np.array([4.0, 0.0, 0.0])
    >>> priorlow = np.array([0.1, 0.0, 0.0])
    >>> priorup  = np.array([0.1, 0.0, 0.0])
    >>> output = mc3.fit(data, uncert, quad, params, indparams=[x],
            prior=prior, priorlow=priorlow, priorup=priorup)
    >>> print(output['bestp'], output['best_chisq'], output['best_log_post'], sep='\n')
    [ 4.01743461 -2.00989433  0.45686521]
    93.77082119449915
    93.80121777303248

.. py:function:: mcmc(*args, **kwargs)
.. code-block:: pycon

    This function has been deprecated. Use mc3.sample() instead.


mc3.plots
_________


.. py:module:: mc3.plots

.. py:function:: trace(posterior, zchain=None, pnames=None, thinning=1, burnin=0, fignum=100, savefile=None, fmt='.', ms=2.5, fs=11)
.. code-block:: pycon

    Plot parameter trace MCMC sampling.

    Parameters
    ----------
    posterior: 2D float ndarray
        An MCMC posterior sampling with dimension: [nsamples, npars].
    zchain: 1D integer ndarray
        the chain index for each posterior sample.
    pnames: Iterable (strings)
        Label names for parameters.
    thinning: Integer
        Thinning factor for plotting (plot every thinning-th value).
    burnin: Integer
        Thinned burn-in number of iteration (only used when zchain is not None).
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

.. py:function:: histogram(posterior, pnames=None, thinning=1, fignum=300, savefile=None, bestp=None, quantile=None, pdf=None, xpdf=None, ranges=None, axes=None, lw=2.0, fs=11, percentile=None)
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
    quantile: Float
        If not None, plot the quantile- highest posterior density region
        of the distribution.  For example, set quantile=0.68 for a 68% HPD.
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

    Deprecated Parameters
    ---------------------
    percentile: Float
        Deprecated. Use quantile instead.

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

  '/home/pcubillos/Dropbox/IWF/projects/2014_mc3/multiproc/MCcubed/'

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

.. py:function:: burn(Zdict=None, burnin=None, Z=None, zchain=None, sort=True)
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
    zchain: 1D integer ndarray
        Chain indices for the samples in Z (used only of Zdict is None).
    sort: Bool
        If True, sort the outputs by chain index.

    Returns
    -------
    posterior: 2D float ndarray
        Burned posterior distribution.
    zchain: 1D integer ndarray
        Burned zchain array.
    zmask: 1D integer ndarray
        Indices that transform Z into posterior.

    Examples
    --------
    >>> import mc3.utils as mu
    >>> import numpy as np
    >>> # Mock a posterior-distribution output:
    >>> Z = np.expand_dims([0., 1, 10, 20, 30, 11, 31, 21, 12, 22, 32], axis=1)
    >>> zchain = np.array([-1, -1, 0, 1, 2, 0, 2, 1, 0, 1, 2])
    >>> Zdict = {'posterior':Z, 'zchain':zchain, 'burnin':1}
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
    >>> posterior, zchain, zmask = mu.burn(Z=Z, zchain=zchain, burnin=1)
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

.. py:function:: log_prior(posterior, prior, priorlow, priorup, pstep)
.. code-block:: pycon

    Compute -2*log(prior) for a given sample.

    This is meant to be the weight added by the prior to chi-square
    when optimizing a Bayesian posterior.  Therefore, there is a
    constant offset with respect to the true -2*log(prior) that can
    be neglected.

    Parameters
    ----------
    posterior: 1D/2D float ndarray
        A parameter sample of shape [nsamples, nfree].
    prior: 1D ndarray
        Parameters priors.  The type of prior is determined by priorlow
        and priorup:
            Gaussian: if both priorlow>0 and priorup>0
            Uniform:  else
        The free parameters in prior must correspond to those
        parameters contained in the posterior, i.e.:
        len(prior[pstep>0]) = nfree.
    priorlow: 1D ndarray
        Lower prior uncertainties.
    priorup: 1D ndarray
        Upper prior uncertainties.
    pstep: 1D ndarray
        Parameter masking determining free (pstep>0), fixed (pstep==0),
        and shared parameters.

    Returns
    -------
    logp: 1D float ndarray
        Sum of -2*log(prior):
        A uniform prior returns     logp = 0.0
        A Gaussian prior returns    logp = (param-prior)**2/prior_uncert**2
        A log-uniform prior returns logp = -2*log(1/param)

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np

    >>> # A posterior of three samples and two free parameters:
    >>> post = np.array([[3.0, 2.0],
    >>>                  [3.1, 1.0],
    >>>                  [3.6, 1.5]])

    >>> # Trivial case, uniform priors:
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.0, 0.0])
    >>> priorup  = np.array([0.0, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [0. 0. 0.]

    >>> # Gaussian prior on first parameter:
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.1, 0.0])
    >>> priorup  = np.array([0.1, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [25. 16. 1.]

    >>> # Posterior comes from a 3-parameter model, with second fixed:
    >>> prior    = np.array([3.5, 0.0, 0.0])
    >>> priorlow = np.array([0.1, 0.0, 0.0])
    >>> priorup  = np.array([0.1, 0.0, 0.0])
    >>> pstep    = np.array([1.0, 0.0, 1.0])
    >>> log_prior = ms.log_prior(post, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    [25. 16. 1.]

    >>> # Also works for a single 1D params array:
    >>> params   = np.array([3.0, 2.0])
    >>> prior    = np.array([3.5, 0.0])
    >>> priorlow = np.array([0.1, 0.0])
    >>> priorup  = np.array([0.1, 0.0])
    >>> pstep    = np.array([1.0, 1.0])
    >>> log_prior = ms.log_prior(params, prior, priorlow, priorup, pstep)
    >>> print(log_prior)
    25.0

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

    Deprecated Parameters
    ---------------------
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

