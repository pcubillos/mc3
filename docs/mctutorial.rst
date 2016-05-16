.. _mctutorial:

MCMC Tutorial
=============

This tutorial describes the available options when running an MCMC with ``MC3``.
As said before, the MCMC can be run from the shell prompt or through a function call in the Python interpreter.

Argument Inputs
---------------

When running from the shell, the arguments can be input as command-line
arguments.  To see all the available options, run:

.. code-block:: shell

   ./mc3.py --help

When running from a Python interactive session, the arguments can be input
as function arguments.  To see the available options, run:

.. code-block:: python

   import MCcubed as mc3
   help(mc3.mcmc)

Additionally (and strongly recommended),
whether you are running the MCMC from the shell or from
the interpreter, the arguments can be input through a configuration file.

Configuration Files
-------------------

The ``MC3`` configuration file follows the `ConfigParser <https://docs.python.org/2/library/configparser.html>`_ format.
The following code block shows an example for an MC3 configuration file:

.. code-block:: python

  # Comment lines (like this one) are allowed and ignored
  # Strings don't need quotation marks
  [MCMC]
  # DEMC general options:
  nsamples  = 1e5
  burnin    = 1000
  nchains   = 7
  walk      = snooker
  # Fitting function:
  func      = quad quadratic ../MCcubed/examples/models
  # Model inputs:
  params    = params.dat
  indparams = indp.npz
  # The data and uncertainties:
  data      = data.npz

MCMC-run Configuration
----------------------

This example describes the basic MCMC argument configuration.
The following sub-sections make up a script meant to be run from the Python
interpreter.  The complete example script is located at `tutorial01 <https://github.com/pcubillos/MCcubed/blob/master/examples/tutorial01/tutorial01.py>`_.


Data and Data Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``data`` argument (required) defines the dataset to be fitted.
This argument can be either a 1D float ndarray or the filename (a string)
where the data array is located.

The ``uncert`` argument (required) defines the :math:`1\sigma` uncertainties
of the ``data`` array.
This argument can be either a 1D float ndarray (same length of ``data``) or the filename where the data uncertainties are located.

.. code-block:: python

   # Create a synthetic dataset using a quadratic polynomial curve:
   import sys
   import numpy as np
   sys.path.append("../MCcubed/examples/models/")
   from quadratic import quad

   x  = np.linspace(0, 10, 1000)         # Independent model variable
   p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
   y  = quad(p0, x)                      # Noiseless model
   uncert = np.sqrt(np.abs(y))           # Data points uncertainty
   error  = np.random.normal(0, uncert)  # Noise for the data
   data   = y + error                    # Noisy data set

.. note:: See the :ref:`datafile` Section below to find out how to set ``data`` and ``uncert`` as a filename.


Modeling Function
^^^^^^^^^^^^^^^^^

The ``func`` argument (required) defines the parameterized modeling function.
The user can set ``func`` either as a callable, e.g.:

.. code-block:: python

   # Define the modeling function as a callable:
   sys.path.append("../MCcubed/examples/models/")
   from quadratic import quad
   func = quad

or as a tuple of strings pointing to the modeling function, e.g.:

.. code-block:: python

   # A three-elements tuple indicates the function name, the module
   # name (without the '.py' extension), and the path to the module.
   func = ("quad", "quadratic", "../MCcubed/examples/models/")

   # Alternatively, if the module is already within the scope of the
   # Python path, the user can set func with a two-elements tuple:
   sys.path.append("../MCcubed/examples/models/")
   func = ("quad", "quadratic")

.. .. important::
.. note:: Important!

   The only requirement for the modeling function is that its arguments follow
   the same structure of the callable in ``scipy.optimize.leastsq``, i.e.,
   the first argument contains the list of fitting parameters.

The ``indparams`` argument (optional) packs any additional argument that the
modeling function may require:

.. code-block:: python

   # indparams contains additional arguments of func (if necessary). Each
   # additional argument is an item in the indparams tuple:
   indparams = [x]

.. note::

   Even if there is only one additional argument to ``func``, indparams must
   be defined as a tuple (as in the example above).  Eventually, the modeling
   function could be called with the following command:

   ``model = func(params, *indparams)``

Fitting Parameters
^^^^^^^^^^^^^^^^^^

The ``params`` argument (required) contains the initial-guess values for the model fitting parameters.  The ``params`` argument must be a 1D float ndarray.

.. code-block:: python

   # Array of initial-guess values of fitting parameters:
   params   = np.array([ 10.0,  -2.0,   0.1])

The ``pmin`` and ``pmax`` arguments (optional) set the lower and upper boundaries explored by the MCMC for each fitting parameter.

.. code-block:: python

   # Lower and upper boundaries for the MCMC exploration:
   pmin     = np.array([-10.0, -20.0, -10.0])
   pmax     = np.array([ 40.0,  20.0,  10.0])

If a proposed step falls outside the set boundaries,
that iteration is automatically rejected.
The default values for each element of ``pmin`` and ``pmax`` are
``-np.inf`` and ``+np.inf``, respectively.
The ``pmin`` and ``pmax`` arrays must have the same size of ``params``.

Stepsize, Fixed, and Shared Paramerers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``stepsize`` argument (required) is a 1D float ndarray,
where each element correspond to one of the fitting parameters.

.. code-block:: python

   stepsize = np.array([  1.0,   0.5,   0.1])

The stepsize has a dual purpose: (1) detemines the free, fixed, and
shared parameters; and (2) determines the step size of proposal jumps.

To fix a parameter at the given initial-guess value,
set the stepsize of the given parameter to :math:`0`.
To share the same value for multiple parameters along the MCMC exploration,
set the stepsize of the parameter equal to the negative
index of the sharing parameter, e.g.:

.. code-block:: python

   # If I want the second, third, and fourth model parameters to share the same value:
   stepsize = np.array([1.0, 3.0, -2, -2])

.. note::

   Clearly, in the current example it doesn't make sense to share parameter
   values.  However, for an eclipe model for example, one may want to share
   the ingress and egress times.

Additionally, when ``walk='mrw'`` (see :ref:`walk` section), ``stepsize``
sets the standard deviation, :math:`\sigma`, of the Gaussian proposal jump for
the given parameter (see Eq. :eq:`gaussprop`).

Lastly, ``stepsize`` sets the standard deviation of the initial sampling
for the chains (see :ref:`mcchains` section).


Parameter Priors
^^^^^^^^^^^^^^^^

The ``prior``, ``priorlow``, and ``priorup`` arguments (optional) set the
prior probability distributions of the fitting parameters.
Each of these arguments is a 1D float ndarray.

.. code-block:: python

   # priorlow defines whether to use uniform non-informative (priorlow = 0.0),
   # Jeffreys non-informative (priorlow < 0.0), or Gaussian prior (priorlow > 0.0).
   # prior and priorup are irrelevant if priorlow <= 0 (for a given parameter)
   prior    = np.array([ 0.0,  0.0,   0.0])
   priorlow = np.array([ 0.0,  0.0,   0.0])
   priorup  = np.array([ 0.0,  0.0,   0.0])

MC3 supports three types of priors.
If a value of ``priorlow`` is :math:`0.0` (default) for a given parameter,
the MCMC will apply a uniform non-informative prior:

.. math::
   p(\theta) = \frac{1}{\theta_{\rm max} - \theta_{\rm min}},
   :label: noninfprior

.. note::

   This is appropriate when there is no prior knowledge of the
   value of :math:`\theta`.


If ``priorlow`` is less than :math:`0.0` for a given parameter,
the MCMC will apply a Jeffreys non-informative prior
(uniform probability per order of magnitude):

.. math::
   p(\theta) = \frac{1}{\theta \ln(\theta_{\rm max}/\theta_{\rm min})},
   :label: jeffreysprior

.. note::

    This is valid only when the parameter takes positive values.
    This is a more appropriate prior than a uniform prior when :math:`\theta`
    can take values over several orders of magnitude.
    For more information, see [Gregory2005]_, Sec. 3.7.1.

.. note::  Practical note!

   In practice, I have seen better results when one fits
   :math:`\log(\theta)` rather than :math:`\theta` with a Jeffreys prior.


Lastly, if ``priorlow`` is greater than  :math:`0.0` for a given parameter,
the MCMC will apply a Gaussian informative prior:

.. math::
   p(\theta) = \frac{1}{\sqrt{2\pi\sigma_{p}^{2}}}
          \exp\left(\frac{-(\theta-\theta_{p})^{2}}{2\sigma_{p}^{2}}\right),
   :label: gaussianprior

where ``prior`` sets the prior value :math:`\theta_{p}`, and
``priorlow`` and ``priorup``
set the lower and upper :math:`1\sigma` prior uncertainties,
:math:`\sigma_{p}`, of the prior (depending if the proposed value
:math:`\theta` is lower or higher than :math:`\theta_{p}`).

.. note::

   Note that, even when the parameter boundaries are not known or when
   the parameter is unbound, this prior is suitable for use in the MCMC
   sampling, since the proposed and current state priors divide out in
   the Metropolis ratio.


.. _walk:

Random Walk
^^^^^^^^^^^

The ``walk`` argument (optional) defines which random-walk algorithm
for the MCMC:

.. code-block:: python

   # Choose between: 'snooker', 'demc', or 'mrw':
   walk = 'snooker'

If ``walk = 'snooker'`` (default, recommended), ``MC3`` will use the
DEMC-z algorithm with snooker propsals (see [BraakVrugt2008]_).
If ``walk = 'demc'``, ``MC3`` will use Differential-Evolution
MCMC algorithm (see [terBraak2006]_).

If ``walk = 'mrw'``, ``MC3`` will use the classical Metropolis-Hastings
algorithm with Gaussian proposal distributions.  I.e., in each
iteration and for each parameter, :math:`\theta`, the MCMC will propose
jumps, drawn from
Gaussian distributions centered at the current value, :math:`\theta_0`, with
a standard deviation, :math:`\sigma`, given by the values in the ``stepsize``
argument:

.. math::
   q(\theta) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(\theta-\theta_0)^2}{2 \sigma^2}\right)
   :label: gaussprop

.. _mcchains:

MCMC Chains Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The following arguments set the MCMC chains configuration:

.. code-block:: python

   nsamples =  1e5     # Number of MCMC samples to compute
   nchains  =    7     # Number of parallel chains
   burnin   = 1000     # Number of burned-in samples per chain
   thinning =    1     # Thinning factor for outputs

   # Distribution for the initial samples:
   kickoff = 'normal'  # Choose between: 'normal' or  'uniform'
   hsize = 10          # Number of initial samples per chain

``MC3`` automatically runs in multiple processors, assigning one CPU per chain.
Additionaly, the central MCMC hub will use one extra CPU.  Thus, the total
number of CPUs used is ``nchains + 1``.

The ``nsamples`` argument (optional, float, default=1e5) sets the total
number of samples to compute.

The ``nchains`` argument (optional, integer, default=7) sets the number
of parallel chains to use.  The number of iterations run for each chain
will be ``ceil(nsamples/nchains)``.

.. note:: For ``walk='snooker'``, an MCMC works well from 
    3 chains.  For ``walk='demc'``, [terBraak2006]_ suggest using
    :math:`2d` chains, with :math:`d` the number of free parameters.

The ``burnin`` argument (optional, integer, default=0) sets the number
of burned-in (removed) iterations at the beginning of each chain.

The ``thinning`` argument (optional, integer, default=1) sets the chains
thinning factor (discarding all but every ``thinning``-th sample).
To reduce the memory usage, when requested, only the thinned samples
are stored (and returned).

.. note:: Thinning is often unnecessary for a DE run, since this algorithm
          reduces significatively the sampling autocorrelation.

To set the starting point of the MCMC chains, ``MC3`` draws samples either
from a normal (default) or uniform distribution (determined by
the ``kickoff`` argument).  The mean and standard deviation of the normal
distribution are set by the ``params`` and ``stepsize`` arguments,
respectively.
The uniform distribution is constrained between the ``pmin`` and ``pmax``
boundaries.
The ``hsize`` argument determines the size of the starting sample.
All draws from the initial sample are discarded from the returned
posterior distribution.

Optimization
^^^^^^^^^^^^

The ``leastsq`` argument (optional, boolean, default=False) is a flag that
indicates MC3 to run a least-squares optimization before running the MCMC.
MC3 implements the Levenberg-Marquardt algorithm via the
``scipy.optimize.leastsq`` function.

.. note:: The parameter boundaries,  fixed and shared-values, and priors
          setup will apply for the minimization.

The ``chisqscale`` argument (optional, boolean, default=False) is a flag that
indicates MC3 to scale the data uncertainties to force a reduced
:math:`\chi^{2}` equal to :math:`1.0`.  The scaling applies by multiplying all
uncertainties by a common scale factor.

.. code-block:: python

   leastsq    = True   # Least-squares minimization prior to the MCMC
   chisqscale = False  # Scale the data uncertainties such that red. chisq = 1


Gelman-Rubin Convergence Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``grtest`` argument (optional, boolean, default=False) is a flag that
indicates MC3 to run the Gelman-Rubin convergence test for the MCMC sample of
fitting parameters.
Values larger than 1.01 are indicative of non-convergence.
See [GelmanRubin1992]_ for further information.

.. The ``grexit`` argument (optional, boolean, default=False)
   is a flag that allows the MCMC to stop if the Gelman-Rubin test returns
   values below 1.01 for all parameter, two consecutive times.

.. code-block:: python

   grtest  = True   # Calculate the GR convergence test
..   grexit  = False  # Stop the MCMC after two successful GR

.. note:: The Gelman-Rubin test is computed every 10% of the MCMC exploration.


Wavelet-Likelihood MCMC
^^^^^^^^^^^^^^^^^^^^^^^

The ``wlike`` argument (optional, boolean, default=False) allows MC3 to
implement the Wavelet-based method to estimate time-correlated noise.
When using this method, the used must append the three additional fitting
parameters (:math:`\gamma, \sigma_{r}, \sigma_{w}`) from Carter & Winn (2009)
to the end of the ``params`` array.  Likewise, add the correspoding values
to the ``pmin``, ``pmax``, ``stepsize``, ``prior``, ``priorlow``,
and ``priorup`` arrays.
For further information see [CarterWinn2009]_.

.. code-block:: python

   wlike = False  # Use Carter & Winn's Wavelet-likelihood method.

File Outputs
^^^^^^^^^^^^

The following arguments set the output files produced by MC3:

.. code-block:: python

   log       = 'MCMC.log'         # Save the MCMC screen outputs to file
   savefile  = 'MCMC_sample.npz'  # Save the MCMC parameters sample to file
   plots     = True               # Generate best-fit, trace, and posterior plots
   rms       = False              # Compute and plot the time-averaging test
   full_output = False            # Return the full posterior sample
..   savemodel = 'MCMC_models.npz'  # Save the MCMC evaluated models to file

The ``log`` argument (optional, string, default=None)
sets the file name where to store ``MC3``'s screen output.

.. The ``savefile`` and ``savemodel`` arguments (optional, string, default=None)
 set the file names where to store the MCMC parameters sample and evaluated
 models.
 MC3 saves the files as three-dimensional ``.npz`` binary files,
 The first dimension corresponds to the chain index,
 the second dimension the fitting parameter or data point
 (for ``savefile`` and ``savemodel``, respectively),
 and the third dimension the iteration number.

The ``savefile`` arguments (optional, string, default=None)
set the file names where to store the MCMC outputs into a ``.npz`` file, with
keywords ``bestp``, ``Z``, and ``Zchain``.
The files can be read with the ``numpy.load()`` function.
``bestp`` is a 1D array with the best-fitting parameters (including fixed
and shared parameters), ``Z`` is a 2D array (Nsamples, Nfree) containing the
thinned MCMC parameter posterior of the free parameters (excluding
fixed and shared).  This array includes the initial and burnin samples.
``Zchain`` is a 1D array containing the chain index for each sample in ``Z``.

The ``plots`` argument (optional, boolean, default=False) is a flag that
indicates MC3 to generate and store the data (along with the best-fitting
model) plot,
the MCMC-chain trace plot for each parameter,
and the marginalized and pair-wise posterior plots.

The ``rms`` argument (optional, boolean, default=False) is a flag that
indicates ``MC3`` to compute the time-averaging test for time-correlated noise
and generate a rms-vs-binsize plot (see [Winn2008]_).

The ``full_output`` argument (optional, bool, default=False) flags the code
to return the full posterior sampling array (``Z``), including the initial
and burnin samples.  The posterior will still be thinned though.


Returned Values
^^^^^^^^^^^^^^^

When run from a pyhton interactive session, ``MC3`` will return four arrays:
``bestp``, a 1D array with the best-fitting parameters (including fixed and
shared parameters); ``uncert``, a 1D array with the parameter uncertainties
(including that of fixed and shared parameters);
``posterior``, a 2D array containing the burned-in, thinned MCMC sample
of the parameters posterior distribution (with dimensions
[nsamples, nfree], excluding fixed and shared parameters); and 
``Zchain``, a 1D array with the indices of the chains for each sample in
``posterior``.

.. code-block:: python

  # Run the MCMC:
  bestp, uncertp, posterior, Zchain = mc3.mcmc(data=data, uncert=uncert,
      func=func, indparams=indparams,
      params=params, pmin=pmin, pmax=pmax, stepsize=stepsize,
      prior=prior, priorlow=priorlow, priorup=priorup,
      walk=walk, nsamples=nsamples,  nchains=nchains,
      burnin=burnin, thinning=thinning,
      leastsq=leastsq, chisqscale=chisqscale,
      hsize=hsize, kickoff=kickoff,
      grtest=grtest, wlike=wlike, log=log,
      plots=plots, savefile=savefile, rms=rms, full_output=full_output)

.. note::  Note that since bestp and uncertp include the values for all
  model parameters, including fixed and shared parameters. Thus, the dimensions
  of the posterior array may not match.

Resume a previous MC3 Run
^^^^^^^^^^^^^^^^^^^^^^^^^

TBD

Inputs from Files
-----------------

The ``data``, ``uncert``, ``indparams``, ``params``, ``pmin``, ``pmax``,
``stepsize``, ``prior``, ``priorlow``, and ``priorup`` input arrays
can be optionally be given as input file.
Furthermore, multiple input arguments can be combined into a single file.

.. _datafile:

Data
^^^^

The ``data``, ``uncert``, and ``indparams`` inputs can be provided as
binary ``numpy`` ``.npz`` files.
``data`` and ``uncert`` can be stored together into a single file.
An ``indparams`` input file contain the list of independent variables
(must be a list, even if there is a single independent variable).

The ``utils`` sub-package of ``MC3`` provide utility functions to
save and load these files.
The ``preamble.py`` file in
`demo02 <https://github.com/pcubillos/MCcubed/blob/master/examples/demo02/>`_
gives an example of how to create ``data`` and ``indparams`` input files:

.. code-block:: python

  # Import the necessary modules:
  import sys
  import numpy as np

  # Import the modules from the MCcubed package:
  sys.path.append("../MCcubed/")
  import MCcubed as mc3
  sys.path.append("../MCcubed/examples/models/")
  from quadratic import quad


  # Create a synthetic dataset using a quadratic polynomial curve:
  x  = np.linspace(0.0, 10, 1000)       # Independent model variable
  p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
  y  = quad(p0, x)                      # Noiseless model
  uncert = np.sqrt(np.abs(y))           # Data points uncertainty
  error  = np.random.normal(0, uncert)  # Noise for the data
  data   = y + error                    # Noisy data set

  # data.npz contains the data and uncertainty arrays:
  mc3.utils.savebin([data, uncert], 'data.npz')
  # indp.npz contains a list of variables:
  mc3.utils.savebin([x], 'indp.npz')


Fitting Parameters
^^^^^^^^^^^^^^^^^^

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

  # To run MCMC, set the arguments to the file names:
  data      = 'data.npz'
  indparams = 'indp.npz'
  params    = 'params.txt'
  # Run MCMC:
  bestp, uncertp, posterior, Zchain = mc3.mcmc(data=data, func=func,
      indparams=indparams, params=params,
      walk=walk, nsamples=nsamples,  nchains=nchains,
      burnin=burnin, thinning=thinning,
      leastsq=leastsq, chisqscale=chisqscale,
      hsize=hsize, kickoff=kickoff,
      grtest=grtest, wlike=wlike, log=log,
      plots=plots, savefile=savefile, rms=rms, full_output=full_output)



References
----------

.. [CarterWinn2009] `Carter & Winn (2009): Parameter Estimation from Time-series Data with Correlated Errors: A Wavelet-based Method and its Application to Transit Light Curves <http://adsabs.harvard.edu/abs/2009ApJ...704...51C>`_
.. [GelmanRubin1992] `Gelman & Rubin (1992): Inference from Iterative Simulation Using Multiple Sequences <http://projecteuclid.org/euclid.ss/1177011136>`_
.. [Gregory2005] `Gregory (2005): Bayesian Logical Data Analysis for the Physical Sciences <http://adsabs.harvard.edu/abs/2005blda.book.....G>`_
.. [terBraak2006] `ter Braak (2006): A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution <http://dx.doi.org/10.1007/s11222-006-8769-1>`_
.. [BraakVrugt2008] `ter Braak & Vrugt (2008): Differential Evolution Markov Chain with snooker updater and fewer chains <http://dx.doi.org/10.1007/s11222-008-9104-9>`_
.. [Winn2008] `Winn et al. (2008): The Transit Light Curve Project. IX. Evidence for a Smaller Radius of the Exoplanet XO-3b <http://adsabs.harvard.edu/abs/2008ApJ...683.1076W>`_
