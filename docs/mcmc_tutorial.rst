.. _mctutorial:

MCMC Tutorial
=============

This tutorial describes the available options when running an MCMC
with ``MC3``.  The following sections make up a script meant to be run
from the Python interpreter or in a Python script.  At the
:ref:`bottom of this page <mcmc-run>` you can see the entire script.

.. _preamble:

Preamble
--------

In this tutorial, we will use the following function to create and fit
a synthetic dataset following a quadratic behavior:

.. literalinclude:: ../examples/tutorial.py
  :lines: 6-17


Argument Inputs
---------------

From the shell, the arguments can be input as command-line arguments.
However, in this case, the best option is to specify all inputs in a
cconfiguration file.  An ``MC3`` configuration file follows the
``configparser`` standard format described `here
<https://docs.python.org/2/library/configparser.html>`_.
To see all the available options, run:

.. code-block:: shell

    mc3 --help

From the Python interpreter, the arguments must be input as function
arguments.  To see the available options, run:

.. code-block:: python

    import mc3
    help(mc3.sample)

.. _data:

Input Data
----------

The ``data`` and ``uncert`` arguments (required) defines the dataset
to be fitted and their :math:`1\sigma` uncertainties, respectively.
Each one of these arguments is a 1D float ndarray:

.. literalinclude:: ../examples/tutorial.py
  :lines: 20-29

.. note:: Alternatively, the ``data`` argument can be a string
          specifying a Numpy npz filename containing the data and
          uncert arrays. See the :ref:`datafile` Section below.

.. _func:

Modeling Function
-----------------

The ``func`` argument (required) defines the parameterized modeling
function fitting the data.  The user can either set ``func`` as a
callable, e.g.:

.. literalinclude:: ../examples/tutorial.py
    :lines: 32-33

or as a tuple of strings pointing to the modeling function, e.g.:

.. code-block:: python

    # A three-element tuple indicates the function's name, the module
    # name (without the '.py' extension), and the path to the module.
    func = ('quad', 'quadratic', '/path/to/quadratic/model/')

    # If the module is already within the scope of the Python path,
    # the user can set func as a two-elements tuple:
    func = ('quad', 'quadratic')

.. note:: The only requirement for the modeling function is that its
    arguments follow the same structure of the callable in
    ``scipy.optimize.leastsq``, i.e., the first argument is a 1D
    iterable containing the fitting parameters.

The ``indparams`` argument (optional) contains any additional argument
required by ``func``:

.. literalinclude:: ../examples/tutorial.py
  :lines: 35-36

.. note:: Even if there is only one additional argument to ``func``,
    ``indparams`` must be defined as a tuple (as in the example
    above).  Eventually, the modeling function has to able to be
    called as: ``model = func(params, *indparams)``


Fitting Parameters
------------------

The ``params`` argument (required) is a 1D float ndarray containing
the initial-guess values for the model fitting parameters.

.. literalinclude:: ../examples/tutorial.py
  :lines: 38-39

The ``pmin`` and ``pmax`` arguments (optional) are 1D float ndarrays
that set lower and upper boundaries explored by the MCMC, for each
fitting parameter (same size as ``params``).

.. literalinclude:: ../examples/tutorial.py
  :lines: 40-42

If a proposed step falls outside the set boundaries,
that iteration is automatically rejected.
The default values for each element of ``pmin`` and ``pmax`` are
``-np.inf`` and ``+np.inf``, respectively.


Parameters Stepping Behavior
----------------------------

The ``pstep`` argument (optional) is a 1D float ndarray that defines
the stepping behavior of the fitting parameters over the parameter
space.  This argument has actually a dual purpose:

.. _behavior:

Stepping Behavior
~~~~~~~~~~~~~~~~~

First, it can keep a fitting parameter fixed by setting its ``pstep``
value to zero, for example:

.. code-block:: python

    # Keep the third parameter fixed:
    pstep = np.array([1.0, 0.5, 0.0])

It can force a fitting parameter to share its value with another
parameter by setting its ``pstep`` value equal to the negative index
of the sharing parameter, for example:

.. code-block:: python

    # Make the third parameter share the value of the second parameter:
    pstep = np.array([1.0, 0.5, -2])

Otherwise, a positive ``pstep`` value leaves the parameter as a free
fitting parameter:

.. literalinclude:: ../examples/tutorial.py
  :lines: 43-44



Stepping Scale
~~~~~~~~~~~~~~

``pstep`` also sets the step size of the free parameters.  For a
differential-evolution run (e.g., ``sampler = 'snooker'``), ``MC3``
starts the MCMC drawing samples from a normal distribution for each
parameter, whose standard deviation is set by the ``pstep`` values.
For a classic Metropolis random walk (``sampler = 'mrw'``), the ``pstep``
values set the standard deviation of the Gaussian proposal jumps for
each parameter.

For more details on the MCMC algorithms, see :ref:`sampler`.


.. _priors:

Parameter Priors
----------------

The ``prior``, ``priorlow``, and ``priorup`` arguments (optional) are
1D float ndarrays that set the prior estimate, lower uncertainty, and
upper uncertainty of the fitting parameters.  ``MC3`` supports two
types of priors:

A ``priorlow`` value of zero (default) defines a uniform prior between
the parameter boundaries.  This is appropriate when there is no prior
knowledge for a parameter :math:`\theta`:

.. math::
    p(\theta) = \frac{1}{\theta_{\rm max} - \theta_{\rm min}},


.. A negative ``priorlow`` value defines a Jeffreys non-informative prior
   (uniform probability per order of magnitude) for a parameter :math:`\theta`:

   .. math::
   p(\theta) = \frac{1}{\theta \ln(\theta_{\rm max}/\theta_{\rm min})},

   This is appropriate when :math:`\theta` can take values over several
   orders of magnitude, and when the parameter takes positive values.
   For more information, see [Gregory2005]_, Sec. 3.7.1.

   .. note:: In practice, for these cases, I have seen better results
          when one fits :math:`\log(\theta)` rather than
          :math:`\theta` with a Jeffreys prior.


Positive ``priorlow`` and ``priorup`` values define a Gaussian
prior for a parameter :math:`\theta`:

.. math::
   p(\theta) = A
          \exp\left(\frac{-(\theta-\theta_{p})^{2}}{2\sigma_{p}^{2}}\right),

where ``prior`` sets the prior value :math:`\theta_{p}`, and
``priorlow`` and ``priorup``
set the lower and upper :math:`1\sigma` prior uncertainties,
:math:`\sigma_{p}`, of the prior (depending if the proposed value
:math:`\theta` is lower or higher than :math:`\theta_{p}`, respectively).
The leading factor is given by: :math:`A =
2/(\sqrt{2\pi}(\sigma_{\rm up}+\sigma_{\rm lo}))` (see [Wallis2014]_), which reduces to
the familiar Gaussian normal distribution when :math:`\sigma_{\rm up}
= \sigma_{\rm lo}`:

.. math::
   p(\theta) = \frac{1}{\sqrt{2\pi}\sigma_{p}}
          \exp\left(\frac{-(\theta-\theta_{p})^{2}}{2\sigma_{p}^{2}}\right),

.. Note that, even when the parameter boundaries are not known or when
   the parameter is unbound, this prior is suitable for use in the MCMC
   sampling, since the proposed and current state priors divide out in
   the Metropolis ratio.

For example, to explicitly set uniform priors for all parameters:

.. literalinclude:: ../examples/tutorial.py
  :lines: 46-49

.. _pnames:

Parameter Names
---------------

The ``pnames`` argument (optional) define the names of the model
parametes to be shown in the scren output and figure labels.  The
screen output will display up to 11 characters.  For figures, the
``texnames`` argument (optional) enables names using LaTeX syntax, for
example:

.. literalinclude:: ../examples/tutorial.py
  :lines: 51-53

If ``texnames = None``, it defaults to ``pnames``. If ``pnames =
None``, it defaults to ``texnames``.  If both arguments are ``None``,
they default to a generic ``[Param 1, Param 2, ...]`` list.

.. _sampler:

Sampler Algorithm
-----------------

The ``sampler`` argument (required) defines the sampler algorithm
for the MCMC:

.. literalinclude:: ../examples/tutorial.py
  :lines: 55-56

The standard Differential-Evolution MCMC algorithm (``sampler = 'demc'``,
[terBraak2006]_) proposes for each chain :math:`i` in state
:math:`\mathbf{x}_{i}`:

.. math:: \mathbf{x}^* = \mathbf{x}_i
                       + \gamma (\mathbf{x}_{R1}-\mathbf{x}_{R2}) + \mathbf{e},

where :math:`\mathbf{x}_{R1}` and :math:`\mathbf{x}_{R2}` are randomly
selected without replacement from the population of current states
except :math:`\mathbf{x}_{i}`.  This implementation adopts
:math:`\gamma=f_{\gamma} 2.38/\sqrt{2 N_{\rm free}}`, with :math:`N_{\rm
free}` the number of free parameters; and
:math:`\mathbf{e}\sim \mathcal{N}(0, \sigma^2)`, with :math:`\sigma=f_{e}`
``pstep``, where the scaling factors are defaulted to
:math:`f_{\gamma}=1.0` and :math:`f_{e}=0.0` (see :ref:`fine-tuning`).

If ``sampler = 'snooker'`` (recommended), ``MC3`` will use the
DEMC-zs algorithm with snooker propsals (see [terBraakVrugt2008]_).

If ``sampler = 'mrw'``, ``MC3`` will use the classical Metropolis-Hastings
algorithm with Gaussian proposal distributions.  I.e., in each
iteration and for each parameter, :math:`\theta`, the MCMC will propose
jumps, drawn from
Gaussian distributions centered at the current value, :math:`\theta_0`, with
a standard deviation, :math:`\sigma`, given by the values in the ``pstep``
argument:

.. math::

   q(\theta) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(\theta-\theta_0)^2}{2 \sigma^2}\right)

.. note:: For ``sampler=snooker``, an MCMC works well with 3 chains or
          more.  For ``sampler=demc``, [terBraak2006]_ suggest using
          :math:`2 N_{\rm free}` chains.  From experience, I recommend
          the ``snooker``, as it is more efficient than most others
          MCMC random walks.


MCMC Configuration
------------------

The following arguments set the MCMC chains configuration:

.. literalinclude:: ../examples/tutorial.py
    :lines: 58-63

The ``nsamples`` argument (required for MCMC runs) sets the total
number of MCMC samples to compute.

The ``burnin`` argument (optional, default: 0) sets the number
of burned-in (removed) iterations at the beginning of each chain.

The ``nchains`` argument (optional, default: 7) sets the number
of parallel chains to use.

The ``ncpu`` argument (optional, default: ``nchains``) sets the number
CPUs to use for the chains.  ``MC3`` runs in multiple processors
through the ``mutiprocessing`` Python Standard-Library package
(additionaly, the central MCMC hub will use one extra CPU.  Thus, the
total number of CPUs used is ``ncpu`` + 1).

.. note:: If ``ncpu+1`` is greater than the number of available CPUs
          in the machine, ``MC3`` will cap ``ncpu`` to the number of
          available CPUs minus one.  To keep a good balance, I
          recommend setting ``nchains`` equal to a multiple of chains
          ``ncpu`` as in the example above.

The ``thinning`` argument (optional, default: 1) sets the chains
thinning factor (discarding all but every ``thinning``-th sample).
To reduce the memory usage, when requested, only the thinned samples
are stored (and returned).

.. note:: Thinning is often unnecessary for a DE run, since this algorithm
          reduces significatively the sampling autocorrelation.


Pre-MCMC Setup
~~~~~~~~~~~~~~

The following arguments set how the code set the initial values for
the MCMC chains:

.. literalinclude:: ../examples/tutorial.py
  :lines: 65-68

The starting point of the MCMC chains come from a random draw, set by
the ``kickoff`` argument (optional, default: 'normal').  This can be a
Normal-distribution draw centered at ``params`` with standard
deviation ``pstep``; or it can be a uniform draw bewteen ``pmin`` and
``pmax``.

The snooker DEMC, in particular, needs an initial sample, set by the
``hsize`` argument (optional, default: 10).  The draws from this
initial sample do not count for the posterior-distribution statistics.

Usually, these variables do not have a significant impact in the
outputs. Thus, they can be left at their default values.

.. _optimization:

Optimization
------------

When not None, the ``leastsq`` argument (optional, default: None) run
a least-squares optimization before the MCMC:

.. literalinclude:: ../examples/tutorial.py
    :lines: 70-72

Set ``leastsq='lm'`` to
use the Levenberg-Marquardt algorithm via `Scipy's leastsq
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq>`_,
or set ``leastsq='trf'`` to use the Trust Region Reflective algorithm
via `Scipy's least_squares
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares>`_.
Fixed and shared-values apply during the optimization (see
:ref:`behavior`), as well as the priors (see :ref:`priors`).

.. note:: From the `scipy
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares>`_
          documentation: Levenberg-Marquardt '*doesn't handle bounds*'
          but is '*the most efficient method for small unconstrained
          problems*'; whereas the Trust Region Reflective algorithm is
          a '*Generally robust method, suitable for large sparse
          problems with bounds*'.

The ``chisqscale`` argument (optional, default: False) is a flag to
scale the data uncertainties to enforce a reduced :math:`\chi^{2}`
equal to :math:`1.0`.  The scaling applies by multiplying all
uncertainties by a common scale factor.


Convergence
-----------

``MC3`` checks for convergence through the Gelman-Rubin test
([GelmanRubin1992]_):

.. literalinclude:: ../examples/tutorial.py
    :lines: 74-77

The ``grtest`` argument (optional, default: False), when True, runs
the Gelman-Rubin convergence test.  Values larger than 1.01 are
indicative of non-convergence.  See [GelmanRubin1992]_ for further
information.  The Gelman-Rubin test is computed every 10% of the MCMC
exploration.

The ``grbreak`` argument (optional, default: 0.0) sets a convergence
threshold to stop an MCMC when GR drops below ``grbreak``.  Reasonable
values seem to be :math:`{\rm grbreak} \lesssim 1.01`.  The default
behavior is not to break (``grbreak = 0.0``).

The ``grnmin`` argument (optional, default: 0.5) sets a minimum number
of valid samples (after burning and thinning) required for
``grbreak``.  If ``grnmin`` is greater than one, it defines the
minimum number of samples to run before breaking out of the MCMC.  If
``grnmin`` is lower than one, it defines the fraction of the total
samples to run before breaking out of the MCMC.


Wavelet-Likelihood MCMC
-----------------------

The ``wlike`` argument (optional, default: False) allows ``MC3`` to
implement the Wavelet-based method to account for time-correlated noise.
When using this method, the used must append the three additional fitting
parameters (:math:`\gamma, \sigma_{r}, \sigma_{w}`) from [CarterWinn2009]_
to the end of the ``params`` array.  Likewise, add the correspoding values
to the ``pmin``, ``pmax``, ``stepsize``, ``prior``, ``priorlow``,
and ``priorup`` arrays.
For further information see [CarterWinn2009]_.

This tutorial won't use the wavelet method:

.. literalinclude:: ../examples/tutorial.py
    :lines: 87-88


.. _fine-tuning:

Fine-tuning
-----------

The :math:`f_{\gamma}` and :math:`f_{e}` factors scale the DEMC
proposal distributions.

.. code-block:: python

    fgamma   = 1.0  # Scale factor for DEMC's gamma jump.
    fepsilon = 0.0  # Jump scale factor for DEMC's "e" distribution

The default :math:`f_{\gamma} = 1.0` value is set such that the MCMC
acceptance rate approaches 25--40%.  Therefore, most of the time, the
user won't need to modify this.  Only if the acceptance rate is very
low, we recommend to set :math:`f_{\gamma} < 1.0`.  The :math:`f_{e}`
factor sets the jump scale for the :math:`\mathbf e` distribution,
which has to have a small variance compared to the posterior.
For further information, see [terBraak2006]_.

.. _logging:

Logging
-------

If not None, the ``log`` argument (optional, default: None) stores the
screen output into a log file.  ``log`` can either be a string
of the filename where to store the log, or an
``mc3.utils.Log`` object (see `API <https://mc3.readthedocs.io/en/makeover/api.html#mc3.utils.Log>`_).

.. literalinclude:: ../examples/tutorial.py
    :lines: 79-80

.. _outputs:

Outputs
-------

The following arguments set the output files produced by ``MC3``:

.. literalinclude:: ../examples/tutorial.py
    :lines: 82-85


The ``savefile`` argument (optional, default: None) defines an
``.npz`` file names where to store the MCMC outputs.  This file
contains the following key--items:

 - ``posterior``: thinned posterior distribution of shape [nsamples, nfree], including burn-in phase.
 - ``zchain``: chain indices for the posterior samples.
 - ``zmask``: posterior mask to remove the burn-in.
 - ``chisq``: :math:`\chi^2` values for the posterior samples.
 - ``log_post``: log of the posterior for the sample (as defined :ref:`here <fittutorial>`).
 - ``burnin``: number of burned-in samples per chain.
 - ``ifree``: Indices of the free parameters.
 - ``pnames``: Parameter names.
 - ``texnames``: Parameter names in Latex format.
 - ``meanp``: mean of the marginal posteriors for each model parameter.
 - ``stdp``: standard deviation of the marginal posteriors for each model parameter.
 - ``CRlo``: lower boundary of the marginal 68%-highest posterior
   density (the credible region) for each model parameter.
 - ``CRhi``: upper boundary of the marginal 68%-HPD.
 - ``stddev_residuals``: standard deviation of the residuals.
 - ``acceptance_rate``: sample's acceptance rate.
 - ``best_log_post``: optimal log of the posterior in the sample (see :ref:`here <fittutorial>`).
 - ``bestp``: model parameters for the ``best_log_post`` sample.
 - ``best_model``: model evaluated at ``bestp``.
 - ``best_chisq``: :math:`\chi^2` for the ``best_log_post`` sample.
 - ``red_chisq``: reduced chi-squared: :math:`\chi^2/(N_{\rm
   data}-N_{\rm free})` for the ``best_log_post`` sample.
 - ``BIC``: Bayesian Information Criterion: :math:`\chi^2 -N_{\rm
   free} \log(N_{\rm data})` for the ``best_log_post`` sample.
 - ``chisq_factor``: Uncertainties scale factor to enforce
   :math:`\chi^2_{\rm red} \equiv 1`.

.. Note:: Notice that if there are fixed or shared parameters, then
          the number of free parameters won't be the same as the
          number of model parameters.  The output posterior ``Z``
          includes only the free parameters, whereas the ``CRlo``,
          ``CRhi``, ``stdp``, ``meanp``, and ``bestp`` outputs include
          all model parameters.


Setting the ``plots`` argument (optional, default: False) to True will
generate data (along with the best-fitting model) plot, the MCMC-chain
trace plot for each parameter, and the marginalized and pair-wise
posterior plots.  Setting the ``ioff`` argument to True (optional,
default: False) will turn the display interactive mode off.

Set the ``rms`` argument (optional, default: False) to True to compute
and plot the time-averaging test for time-correlated noise (see
[Winn2008]_).


.. _mcmc-run:

MCMC Run
--------

Putting it all together, here's a Python script to run an ``MC3``
retrieval explicitly defining all the variables described above:

.. literalinclude:: ../examples/tutorial.py

This routine returns a dictionary containing the outputs listed in
:ref:`outputs`.  The screen output should look like this:

.. code-block:: none

    ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
      Multi-core Markov-chain Monte Carlo (MC3).
      Version 3.0.0.
      Copyright (c) 2015-2019 Patricio Cubillos and collaborators.
      MC3 is open-source software under the MIT license (see LICENSE).
    ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Least-squares best-fitting parameters:
      [ 3.02203328 -2.3897706   0.49543328]

    Yippee Ki Yay Monte Carlo!
    Start MCMC chains  (Thu Aug  8 11:23:20 2019)

    [:         ]  10.0% completed  (Thu Aug  8 11:23:21 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]

    [::        ]  20.0% completed  (Thu Aug  8 11:23:21 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]
    Gelman-Rubin statistics for free parameters:
    [1.02204221 1.02386902 1.02470492]

    [:::       ]  30.0% completed  (Thu Aug  8 11:23:21 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]
    Gelman-Rubin statistics for free parameters:
    [1.00644059 1.00601973 1.00644078]
    All parameters converged to within 1% of unity.

    [::::      ]  40.0% completed  (Thu Aug  8 11:23:22 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]
    Gelman-Rubin statistics for free parameters:
    [1.00332153 1.00383779 1.00326743]
    All parameters converged to within 1% of unity.

    [:::::     ]  50.0% completed  (Thu Aug  8 11:23:22 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]
    Gelman-Rubin statistics for free parameters:
    [1.00286025 1.00297467 1.00258288]
    All parameters converged to within 1% of unity.

    [::::::    ]  60.0% completed  (Thu Aug  8 11:23:22 2019)
    Out-of-bound Trials:
    [0 0 0]
    Best Parameters: (chisq=1035.2269)
    [ 3.02203328 -2.3897706   0.49543328]
    Gelman-Rubin statistics for free parameters:
    [1.00169127 1.0016499  1.0013014 ]
    All parameters converged to within 1% of unity.

    All parameters satisfy the GR convergence threshold of 1.01, stopping
    the MCMC.

    MCMC Summary:
    -------------
      Number of evaluated samples:        60506
      Number of parallel chains:             14
      Average iterations per chain:        4321
      Burned-in iterations per chain:      1000
      Thinning factor:                        1
      MCMC sample size (thinned, burned): 46506
      Acceptance rate:   28.85%

    Param name     Best fit   Lo HPD CR   Hi HPD CR        Mean    Std dev       S/N
    ----------- ----------------------------------- ---------------------- ---------
    y0           3.0220e+00 -1.2142e-01  1.2574e-01  3.0223e+00 1.2231e-01      24.7
    alpha       -2.3898e+00 -7.2210e-02  6.8853e-02 -2.3904e+00 7.0381e-02      34.0
    beta         4.9543e-01 -8.3569e-03  8.9226e-03  4.9557e-01 8.6295e-03      57.4

      Best-parameter's chi-squared:       1035.2269
      Best-parameter's -2*log(posterior): 1035.2269
      Bayesian Information Criterion:     1055.9502
      Reduced chi-squared:                   1.0383
      Standard deviation of residuals:  2.77253

    Output MCMC files:
      'MCMC_tutorial.npz'
      'MCMC_tutorial_trace.png'
      'MCMC_tutorial_pairwise.png'
      'MCMC_tutorial_posterior.png'
      'MCMC_tutorial_RMS.png'
      'MCMC_tutorial_model.png'
      'MCMC_tutorial.log'


------------------------------------------------------------------------


Inputs from Files
-----------------

The ``data``, ``uncert``, ``indparams``, ``params``, ``pmin``, ``pmax``,
``stepsize``, ``prior``, ``priorlow``, and ``priorup`` input arrays
can be optionally be given as input file.
Furthermore, multiple input arguments can be combined into a single file.

.. _datafile:

Input Data File
~~~~~~~~~~~~~~~

The ``data``, ``uncert``, and ``indparams`` inputs can be provided as
binary ``numpy`` ``.npz`` files.
``data`` and ``uncert`` can be stored together into a single file.
An ``indparams`` input file contain the list of independent variables
(must be a list, even if there is a single independent variable).

The ``utils`` sub-package of ``MC3`` provide utility functions to save
and load these files.  This script shows how to create ``data`` and
``indparams`` input files:

.. code-block:: python

  import numpy as np
  import mc3

  # Create a synthetic dataset using a quadratic polynomial curve:
  x  = np.linspace(0.0, 10, 1000)
  p0 = [3, -2.4, 0.5]
  y  = quad(p0, x)
  error  = np.random.normal(0, uncert)

  data   = y + error
  uncert = np.sqrt(np.abs(y))

  # data.npz contains the data and uncertainty arrays:
  mc3.utils.savebin([data, uncert], 'data.npz')
  # indp.npz contains a list of variables:
  mc3.utils.savebin([x], 'indp.npz')


Model Parameters
~~~~~~~~~~~~~~~~

The ``params``, ``pmin``, ``pmax``, ``stepsize``,
``prior``, ``priorlow``, and ``priorup`` inputs
can be provided as plain ASCII files.
For simplycity all of these input arguments can be combined into
a single file.

In the ``params`` file, each line correspond to one model
parameter, whereas each column correspond to one of the input array arguments.
This input file can hold as few or as many of these argument arrays,
as long as they are provided in that exact order.
Empty or comment lines are allowed (and ignored by the reader).
A valid params file look like this:

.. code-block:: none

  #       params            pmin            pmax        stepsize
              10             -10              40             1.0
            -2.0             -20              20             0.5
             0.1             -10              10             0.1

Alternatively, the ``utils`` sub-package of ``MC3`` provide utility
functions to save and load these files:

.. code-block:: python

  params   = [ 10, -2.0,  0.1]
  pmin     = [-10,  -20, -10]
  pmax     = [ 40,   20,  10]
  stepsize = [  1,  0.5,  0.1]

  # Store ASCII arrays:
  mc3.utils.saveascii([params, pmin, pmax, stepsize], 'params.txt')


Then, to run the MCMC simply provide the input file names to the ``MC3``
routine:

.. code-block:: python

    # Set arguments as the file names:
    data      = 'data.npz'
    indparams = 'indp.npz'
    params    = 'params.txt'

    # Run the MCMC:
    mc3_output = mc3.sample(data=data, func=func, params=params,
        indparams=indparams, sampler=sampler, nsamples=nsamples,  nchains=nchains,
        ncpu=ncpu, burnin=burnin, leastsq=leastsq, chisqscale=chisqscale,
        grtest=grtest, grbreak=grbreak, grnmin=grnmin,
        log=log, plots=plots, savefile=savefile, rms=rms)
