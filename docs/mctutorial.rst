.. _mctutorial:

MCMC Tutorial
=============

This tutorial describes the available options when running an MCMC
with ``MC3``.

Argument Inputs
---------------

From the shell, the arguments can be input as command-line arguments.
To see all the available options, run:

.. code-block:: shell

    mc3 --help

However, in this case, the best option is to specify all inputs in a
cconfiguration file.  An ``MC3`` configuration file follows the
``configparser`` standard format described `here
<https://docs.python.org/2/library/configparser.html>`_.


From the Python interpreter, the arguments must be input as function
arguments.  To see the available options, run:

.. code-block:: python

    import mc3
    help(mc3.mcmc)

MCMC Configuration
------------------

This example describes the basic MCMC argument configuration.
The following sub-sections make up a script meant to be run from the Python
interpreter.  The complete example script is located at `tutorial01 <https://github.com/pcubillos/MCcubed/blob/master/examples/tutorial01/tutorial01.py>`_.

Preamble
^^^^^^^^

In this tutorial, we will use the following function to create a
synthetic dataset following a quadratic behavior:

.. code-block:: python

    def quad(p, x):
        """
        Quadratic polynomial function.

        Parameters
            p: Polynomial constant, linear, and quadratic coefficients.
            x: Array of dependent variables where to evaluate the polynomial.
        Returns
            y: Polinomial evaluated at x:  y(x) = p0 + p1*x + p2*x^2
        """
        y = p[0] + p[1]*x + p[2]*x**2.0
        return y


Input Data
^^^^^^^^^^

The ``data`` and ``uncert`` arguments (required) defines the dataset
to be fitted and their :math:`1\sigma` uncertainties, respectively.
Each one of these arguments is a 1D float ndarray.

.. note:: Alternatively, the ``data`` argument can be a string
          specifying a Numpy npz filename containing the data (and
          uncert) array. See the :ref:`datafile` Section below.



.. code-block:: python

   # Create a synthetic dataset using a quadratic polynomial curve:
   import numpy as np

   x  = np.linspace(0, 10, 1000)         # Independent model variable
   p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
   y  = quad(p0, x)                      # Noiseless model
   error  = np.random.normal(0, uncert)  # Noise for the data

   data   = y + error                    # Noisy data set
   uncert = np.sqrt(np.abs(y))           # Data points uncertainty


Modeling Function
^^^^^^^^^^^^^^^^^

The ``func`` argument (required) defines the parameterized modeling
function.  The user can either set ``func`` as a callable, e.g.:

.. code-block:: python

   # Define the modeling function as a callable:
   func = quad

or as a tuple of strings pointing to the modeling function, e.g.:

.. code-block:: python

   # A three-elements tuple indicates the function name, the module
   # name (without the '.py' extension), and the path to the module.
   func = ("quad", "quadratic", "./path/to/quadratic/model/")

   # If the module is already within the scope of the Python path,
   # the user can set func as a two-elements tuple:
   func = ("quad", "quadratic")

.. note::

   The only requirement for the modeling function is that its
   arguments follow the same structure of the callable in
   ``scipy.optimize.leastsq``, i.e., the first argument is a 1D
   iterable containing the fitting parameters.

The ``indparams`` argument (optional) contains any additional argument
required by ``func``:

.. code-block:: python

   # indparams contains additional arguments of func (if necessary). Each
   # additional argument is an item in the indparams tuple:
   indparams = [x]

.. note::

   Even if there is only one additional argument to ``func``, indparams must
   be defined as a tuple (as in the example above).  Eventually, the modeling
   function has to able to be called as:
   ``model = func(params, *indparams)``


Fitting Parameters
^^^^^^^^^^^^^^^^^^

The ``params`` argument (required) is a 1D float ndarray containing
the initial-guess values for the model fitting parameters.

.. code-block:: python

   # Array of initial-guess values of fitting parameters:
   params   = np.array([ 10.0,  -2.0,   0.1])

The ``pmin`` and ``pmax`` arguments (optional) are 1D float ndarrays
that set lower and upper boundaries explored by the MCMC, for each
fitting parameter (same size as ``params``).

.. code-block:: python

   # Lower and upper boundaries for the MCMC exploration:
   pmin     = np.array([-10.0, -20.0, -10.0])
   pmax     = np.array([ 40.0,  20.0,  10.0])

If a proposed step falls outside the set boundaries,
that iteration is automatically rejected.
The default values for each element of ``pmin`` and ``pmax`` are
``-np.inf`` and ``+np.inf``, respectively.


Parameters Stepping Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pstep`` argument (optional) is a 1D float ndarray that defines
the stepping behavior of the fitting parameters over the parameter
space.  This argument has actually a dual purpose:

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

.. code-block:: python

    # Leave all three parameters free:
    pstep = np.array([1.0, 0.5, 0.1])


Stepping Scale
~~~~~~~~~~~~~~

``pstep`` also sets the step size of the free parameters.  For a
differential-evolution run (e.g., ``walk = 'snooker'``), ``MC3``
starts the MCMC drawing samples from a normal distribution for each
parameter, whose standard deviation is set by the ``pstep`` values.
For a classic Metropolis random walk (``walk = 'mrw'``), the ``pstep``
values set the standard deviation of the Gaussian proposal jumps for
each parameter.

For more details on the MCMC algorithms, see :ref:`walk`.


Parameter Priors
^^^^^^^^^^^^^^^^

The ``prior``, ``priorlow``, and ``priorup`` arguments (optional) are
1D float ndarrays that set the prior estimate, lower uncertainty, and
upper uncertainty of the fitting parameters.
``MC3`` supports three types of priors:


.. code-block:: python

   # priorlow defines whether to use uniform non-informative (priorlow = 0.0),
   # Jeffreys non-informative (priorlow < 0.0), or Gaussian prior (priorlow > 0.0).
   # prior and priorup are irrelevant if priorlow <= 0 (for a given parameter)
   prior    = np.array([0.0, 0.0, 0.0])
   priorlow = np.array([0.0, 0.0, 0.0])
   priorup  = np.array([0.0, 0.0, 0.0])

A ``priorlow`` value of zero (default) defines a uniform prior between
the parameter boundaries.  This is appropriate when there is no prior
knowledge for a parameter :math:`\theta`:

.. math::
    p(\theta) = \frac{1}{\theta_{\rm max} - \theta_{\rm min}},


A negative ``priorlow`` value defines a Jeffreys non-informative prior
(uniform probability per order of magnitude) for a parameter :math:`\theta`:

.. math::
   p(\theta) = \frac{1}{\theta \ln(\theta_{\rm max}/\theta_{\rm min})},

This is appropriate when :math:`\theta` can take values over several
orders of magnitude, and when the parameter takes positive values.
For more information, see [Gregory2005]_, Sec. 3.7.1.

.. note::  Practical note!

   In practice, for these cases, I have seen better results when one
   fits :math:`\log(\theta)` rather than :math:`\theta` with a
   Jeffreys prior.


Lastly, a positive ``priorlow`` value defines a Gaussian prior for a
parameter :math:`\theta`:

.. math::
   p(\theta) = \frac{1}{\sqrt{2\pi\sigma_{p}^{2}}}
          \exp\left(\frac{-(\theta-\theta_{p})^{2}}{2\sigma_{p}^{2}}\right),

where ``prior`` sets the prior value :math:`\theta_{p}`, and
``priorlow`` and ``priorup``
set the lower and upper :math:`1\sigma` prior uncertainties,
:math:`\sigma_{p}`, of the prior (depending if the proposed value
:math:`\theta` is lower or higher than :math:`\theta_{p}`).

.. Note that, even when the parameter boundaries are not known or when
   the parameter is unbound, this prior is suitable for use in the MCMC
   sampling, since the proposed and current state priors divide out in
   the Metropolis ratio.


Parameter Names
^^^^^^^^^^^^^^^

The ``pnames`` argument (optional) define the names of the model
parametes to be shown in the scren output and figure labels.  In
figures, the names can use LaTeX syntax.  The screen output will
display up to 11 characters.  Thus, the user can define the
``texnames`` argument (optional), display the appropriate syntax for
screen output and figures, for example:

.. code-block:: python

   pnames   = ["log(alpha)", "beta", "Teff"]
   texnames = [r"$\log(\alpha)$", r"$\beta$", r"$T_{\rm eff}$"]

If ``texnames`` is ``None``, it defaults to ``pnames``. If ``pnames``
is ``None``, it defaults to ``texnames``.  If both arguments are
``None``, they default to a generic ``[Param 1, Param 2, ...]`` list.

.. _walk:

Random Walk
^^^^^^^^^^^

The ``walk`` argument (optional) defines which random-walk algorithm
for the MCMC:

.. code-block:: python

   # Choose between: 'snooker', 'demc', or 'mrw':
   walk = 'snooker'

The standard Differential-Evolution MCMC algorithm (``walk = 'demc'``,
[terBraak2006]_) proposes for each chain :math:`i` in state
:math:`\mathbf{x}_{i}`:

.. math::
   :label: eqdemc

   \mathbf{x}^* = \mathbf{x}_i + \gamma (\mathbf{x}_{R1}-\mathbf{x}_{R2}) + \mathbf{e},

where :math:`\mathbf{x}_{R1}` and :math:`\mathbf{x}_{R2}` are randomly
selected without replacement from the population of current states
without :math:`\mathbf{x}_{i}`.  This implementation adopts
:math:`\gamma=f_{\gamma} 2.38/\sqrt{2 N_{\rm free}}`, and
:math:`\mathbf{e}\sim N(0, f_{e}\,{\rm stepsize})`, with
:math:`N_{\rm free}` the number of free parameters. The scaling factors
are defaulted to :math:`f_{\gamma}=1.0` and :math:`f_{e}=0.0` (see
:ref:`fine-tuning`).

If ``walk = 'snooker'`` (default, recommended), ``MC3`` will use the
DEMC-z algorithm with snooker propsals (see [terBraakVrugt2008]_).

If ``walk = 'mrw'``, ``MC3`` will use the classical Metropolis-Hastings
algorithm with Gaussian proposal distributions.  I.e., in each
iteration and for each parameter, :math:`\theta`, the MCMC will propose
jumps, drawn from
Gaussian distributions centered at the current value, :math:`\theta_0`, with
a standard deviation, :math:`\sigma`, given by the values in the ``stepsize``
argument:

.. math::
   :label: gaussprop

   q(\theta) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(\theta-\theta_0)^2}{2 \sigma^2}\right)

.. note:: For ``walk=snooker``, an MCMC works well from 3 chains.  For
    ``walk=demc``, [terBraak2006]_ suggest using :math:`2*d` chains,
    with :math:`d` the number of free parameters.

I recommend any of the ``snooker`` or ``demc``
algorithms, as they are more efficient than most others MCMC random
walks.  From experience, when deciding between these two, consider
that when the initial guess lays far from the lowest chi-square
region, ``snooker`` seems to produce lower acceptance rates than ideal
(which is solvable setting ``leastsq=True``).  On the other hand,
``demc`` is limited to a high number of chains when there is a high
number of free parameters.


.. _mcchains:

MCMC Config
^^^^^^^^^^^

The following arguments set the MCMC chains configuration:

.. code-block:: python

   nsamples =  1e5     # Number of MCMC samples to compute
   nchains  =    7     # Number of parallel chains
   nproc    =    7     # Number of CPUs to use for chains (default: nchains)
   burnin   = 1000     # Number of burned-in samples per chain
   thinning =    1     # Thinning factor for outputs

   # Distribution for the initial samples:
   kickoff = 'normal'  # Choose between: 'normal' or  'uniform'
   hsize = 10          # Number of initial samples per chain


The ``nsamples`` argument (optional, float, default=1e5) sets the
total number of samples to compute.  The approximate number of
iterations run for each chain will be ``nsamples/nchains``.

The ``nchains`` argument (optional, integer, default=7) sets the number
of parallel chains to use.  The number of iterations run for each chain
will be approximately ``nsamples/nchains``.

``MC3`` runs in multiple processors through the ``mutiprocessing``
package.  The ``nproc`` argument (optional, integer,
default= ``nchains``) sets the number CPUs to use for the chains.
Additionaly, the central MCMC hub will use one extra CPU.  Thus, the
total number of CPUs used is ``nchains + 1``.

.. note:: If ``nproc+1`` is greater than the number of available CPUs
          in the machine (``nCPU``), ``MC3`` will set ``nproc =
          nCPU-1``.  To keep a good balance, I recommend setting
          ``nchains`` equal to a multiple of ``nproc``.


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
indicates ``MC3`` to run a least-squares optimization before running the MCMC.
``MC3`` implements the Levenberg-Marquardt algorithm (``lm=True``) via
``scipy.optimize.leastsq`` or Trust Region Reflective (``lm=False``) via
``scipy.optimize.least_squares``.

.. note:: The parameter boundaries (for TRF only, see :ref:`fittutorial`),
  fixed and shared-values, and priors will apply for the minimization.

The ``chisqscale`` argument (optional, boolean, default=False) is a flag that
indicates ``MC3`` to scale the data uncertainties to force a reduced
:math:`\chi^{2}` equal to :math:`1.0`.  The scaling applies by multiplying all
uncertainties by a common scale factor.

.. code-block:: python

   leastsq    = True   # Least-squares minimization prior to the MCMC
   lm         = True   # Choose Levenberg-Marquardt (True) or TRF algorithm (False)
   chisqscale = False  # Scale the data uncertainties such that red. chisq = 1


Convergence
^^^^^^^^^^^

The ``grtest`` argument (optional, boolean, default=False) is a flag that
indicates MC3 to run the Gelman-Rubin convergence test for the MCMC sample of
fitting parameters.
Values larger than 1.01 are indicative of non-convergence.
See [GelmanRubin1992]_ for further information.

Additionally, the ``grbreak`` argument (optional, boolean,
default=0.0) sets a convergence threshold to stop an MCMC when GR
drops below ``grbreak``.  Reasonable values seem to be ``grbreak``
~1.001--1.005.  The default behavior is not to break (``grbreak=0.0``).

Lastly, the ``grnmin`` argument (optional, integer or float,
default=0.5) sets a minimum number of valid samples (after burning and
thinning) required for ``grbreak``.  If ``grnmin`` is an integer,
require at least ``grnmin`` samples to break out of the MCMC.  If
``grnmin`` is a float (in the range 0.0--1.0), require at least
``grnmin`` times the maximum possible number of valid samples to break
out of the MCMC.

.. code-block:: python

   grtest  = True   # Calculate the GR convergence test
   grbreak = 0.0    # GR threshold to stop the MCMC run
   grnmin  = 0.5    # Minimum fraction or number of samples before grbreak

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

.. _fine-tuning:

Fine-tuning
^^^^^^^^^^^

The :math:`f_{\gamma}` and :math:`f_{e}` factors scale the DEMC
proposal distributions.

.. code-block:: python

   fgamma   = 1.0  # Scale factor for DEMC's gamma jump.
   fepsilon = 0.0  # Jump scale factor for DEMC's "e" distribution

The default :math:`f_{\gamma}=1.0` value is set such that the MCMC
acceptance rate approaches 25-40%.  Therefore, most of the time, the
user won't need to modify this.  Only if the acceptance rate is very
low, we recommend to set :math:`f_{\gamma}<1.0`.  The :math:`f_{e}`
factor sets the jump scale for the :math:`\mathbf e` distribution,
which has to have a small variance compared to the posterior.
For further information see [terBraak2006]_.

.. _file-outputs:

File Outputs
^^^^^^^^^^^^

The following arguments set the output files produced by MC3:

.. code-block:: python

   log       = 'MCMC.log'         # Save the MCMC screen outputs to file
   savefile  = 'MCMC_sample.npz'  # Save the MCMC parameters sample to file
   plots     = True               # Generate best-fit, trace, and posterior plots
   rms       = False              # Compute and plot the time-averaging test
   full_output = False            # Return the full posterior sample
   chireturn = False              # Return chi-square statistics
..   savemodel = 'MCMC_models.npz'  # Save the MCMC evaluated models to file

The ``log`` argument (optional, string, default = ``None``)
sets the file name where to store ``MC3``'s screen output.

.. The ``savefile`` and ``savemodel`` arguments (optional, string, default=None)
 set the file names where to store the MCMC parameters sample and evaluated
 models.
 MC3 saves the files as three-dimensional ``.npz`` binary files,
 The first dimension corresponds to the chain index,
 the second dimension the fitting parameter or data point
 (for ``savefile`` and ``savemodel``, respectively),
 and the third dimension the iteration number.

The ``savefile`` arguments (optional, string, default = ``None``) set
the file names where to store the MCMC outputs into a ``.npz`` file,
with keywords ``bestp``, ``CRlo``, ``CRhi``, ``stdp``, ``meanp``,
``Z``, ``Zchain``, and ``Zchisq``, ``bestchisq``, ``redchisq``,
``chifactor``, ``BIC``, and standard deviation of the residuals ``sdr``.
The files can be read with the
``numpy.load()`` function.  See :ref:`retvals` and the description of
``chireturn`` below for details on the output values.

The ``plots`` argument (optional, boolean, default = ``False``) is a
flag that indicates MC3 to generate and store the data (along with the
best-fitting model) plot, the MCMC-chain trace plot for each
parameter, and the marginalized and pair-wise posterior plots.

The ``rms`` argument (optional, boolean, default = ``False``) is a
flag that indicates ``MC3`` to compute the time-averaging test for
time-correlated noise and generate a rms-vs-binsize plot (see
[Winn2008]_).

The ``full_output`` argument (optional, bool, default = ``False``)
flags the code to return the full posterior sampling array (``Z``),
including the initial and burnin samples.  The posterior will still be
thinned though.

If the ``chireturn`` argument (optional, bool, default = ``False``) is
``True``, ``MC3`` will return an additional tuple containing the
chi-square stats: lowest :math:`\chi^{2}` (``bestchisq``),
:math:`\chi^{2}_{\rm red}` (``redchisq``), scaling factor to enforce
:math:`\chi^{2}_{\rm red} = 1` (``chifactor``), and the Bayesian
Information Criterion BIC (``BIC``).

.. _retvals:

Returned Values
^^^^^^^^^^^^^^^

When run from a pyhton interactive session, ``MC3`` will return a
tuple with six elements (seven if ``chireturn=True``, see above):

- ``bestp``: a 1D array with the best-fitting parameters (including
  fixed and shared parameters).
- ``CRlo``: a 1D array with the lower boundary of the marginal 68%-highest
  posterior density (the credible region) for each parameter,
  with respect to ``bestp``.
- ``CRhi``:a 1D array with the upper boundary of the marginal
  68%-highest posterior density for each parameter, with respect to
  ``bestp``.
- ``stdp``: a 1D array with the standard deviation of the marginal
  posterior for each parameter (including that of fixed and shared
  parameters).
- ``posterior``: a 2D array containing the burned-in, thinned MCMC
  sample of the parameters posterior distribution (with dimensions
  [nsamples, nfree], excluding fixed and shared parameters).
- ``Zchain``: a 1D array with the indices of the chains for each
  sample in ``posterior``.


.. code-block:: python

  # Run the MCMC:
  bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data=data,
      uncert=uncert, func=func, indparams=indparams,
      params=params, pmin=pmin, pmax=pmax, stepsize=stepsize,
      prior=prior, priorlow=priorlow, priorup=priorup,
      walk=walk, nsamples=nsamples,  nchains=nchains,
      nproc=nproc, burnin=burnin, thinning=thinning,
      leastsq=leastsq, lm=lm, chisqscale=chisqscale,
      hsize=hsize, kickoff=kickoff,
      grtest=grtest,  grbreak=grbreak, grnmin=grnmin,
      wlike=wlike, log=log,
      plots=plots, savefile=savefile, rms=rms, full_output=full_output)

.. note:: Note that ``bestp``, ``CRlo``, ``CRhi``, and ``stdp``
  include the values for all model parameters, including fixed and
  shared parameters, whereas ``posterior`` includes only
  the free parameters.  Be careful with the dimesions.

..
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
  bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data=data,
      func=func, indparams=indparams, params=params,
      walk=walk, nsamples=nsamples,  nchains=nchains,
      nproc=nproc, burnin=burnin, thinning=thinning,
      leastsq=leastsq, lm=lm, chisqscale=chisqscale,
      hsize=hsize, kickoff=kickoff,
      grtest=grtest, grbreak=grbreak, grnmin=grnmin,
      wlike=wlike, log=log,
      plots=plots, savefile=savefile, rms=rms, full_output=full_output)
