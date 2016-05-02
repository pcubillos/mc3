.. _fittutorial:

Optimization Tutorial
=====================

The ``MCcubed.fit`` module provides the ``modelfit`` routine for
model-fitting optimization through the least-squares
Levenberg-Marquardt algorith.

``modelfit`` is a wrapper of ``scipy.optimize.leastsq`` with additional
features, including Gaussian-parameter priors, parameter boundaries,
and sharing and fixing parameters.
All ``modelfit`` arguments are identical to those of the MCMC.

Fitting Parameters
^^^^^^^^^^^^^^^^^^

The ``params`` argument (required) contains the initial-guess values
for the model fitting parameters.  The ``params`` argument must be
a 1D float ndarray.

Modeling Function
^^^^^^^^^^^^^^^^^

The ``func`` argument (required) defines the parameterized modeling function.
The only requirement for the modeling function is that its arguments follow
the same structure of the callable in ``scipy.optimize.leastsq``, i.e.,
the first argument contains the list of fitting parameters.

If func requires additional arguments, they can be provided through
the ``indparams`` argument (see :ref:`indp`).
Eventually, the modeling function could be called with the following command:

``model = func(params, *indparams)``


Data and Data Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``data`` argument (required) defines the dataset to be fitted.
This argument can be either a 1D float ndarray or the filename (a string)
where the data array is located.

The ``uncert`` argument (required) defines the :math:`1\sigma` uncertainties
of the ``data`` array.
This argument can be either a 1D float ndarray (same length of ``data``) or the filename where the data uncertainties are located.


.. _indp:

Independent Parameters
^^^^^^^^^^^^^^^^^^^^^^

The ``indparams`` argument (optional) is a tuple (or list) that packs
any additional arguments required by ``func``.
Even if ``indparams`` consists of a single variable, it must be defined
as a list or tuple.


Stepsize: Fixed, and Shared Paramerers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``stepsize`` argument (optional) is a 1D float ndarray,
where each element correspond to one of the fitting parameters.
For optimization, ``stepsize`` determines the free, fixed, and shared
parameters.
If the stepsize is positive (irrelevant of the value), the parameter is
a free fitting parameter.

To fix a parameter at the given initial-guess value,
set the stepsize of the given parameter to :math:`0`.

To copy the value from another parameter (free or fixed),
set the stepsize equal to the negative index of the sharing
parameter.

.. note:: Consider that in this case, contrary to Python standards,
          the indexing starts counting from one instead of zero.  Thus,
          for example, to share a value with that of the first parameter,
          set the parameter's stepsize to :math:`-1`.

Parameter Boundaries
^^^^^^^^^^^^^^^^^^^^

The ``pmin`` and ``pmax`` arguments (optional) are 1D float ndarrays that
set the lower and upper boundaries explored by the minimizer for each
fitting parameter (same size of ``params``).
The default values for each element of ``pmin`` and ``pmax`` are
``-np.inf`` and ``+np.inf``, respectively.

Parameter Priors
^^^^^^^^^^^^^^^^

The ``prior``, ``priorlow``, and ``priorup`` arguments (optional) set the
prior probability distributions of the fitting parameters.
Each of these arguments is a 1D float ndarray.

If a value of ``priorlow`` is :math:`0.0` (default) for a given parameter,
the MCMC will apply a uniform non-informative prior:

.. math::
   p(\theta) = \frac{1}{\theta_{\rm max} - \theta_{\rm min}},
   :label: noninfprior

.. note::

   This is appropriate when there is no prior knowledge of the
   value of :math:`\theta`.


If ``priorlow`` is greater than  :math:`0.0` for a given parameter,
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


Outputs
^^^^^^^

``modelfit`` returns four variables:

- ``chisq`` (float) is the best-fitting chi-square value.
- ``bestparams`` (1D float ndarray) is the array of best-fitting parameters,
  including fixed and shared parameters.
- ``bestmodel`` (1D float ndarray) is the best-fitting model found, i.e.,
    ``func(bestparams, *indparams)``.
- ``lsfit`` (list) is ``scipy.optimize.leastsq``'s full_output return.


Example
^^^^^^^

.. code-block:: python

  import sys
  import MCcubed as mc3  # Add path to mc3 if necessary

  # Get a modeling function (quadractic polynomial):
  sys.path.append("./examples/models/")  # Set the appropriate path
  from quadratic import quad

  # Create a synthetic dataset using a quadratic polynomial curve:
  x  = np.linspace(0, 10, 1000)         # Independent model variable
  p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
  y  = quad(p0, x)                      # Noiseless model
  uncert = np.sqrt(np.abs(y))           # Data points uncertainty
  error  = np.random.normal(0, uncert)  # Noise for the data
  data   = y + error                    # Noisy data set

  # Array of initial-guess values of fitting parameters:
  params   = np.array([ 20.0,  -2.0,   0.1])

  func = quad

  # indparams contains additional arguments of func (besides params):
  indparams = [x]

  params   = np.array([  1.0,   0.0,   0.3])
  stepsize = np.array([  1.0,   1.0,   1.0])  # All model parameters free
  pmin     = np.array([-10.0, -20.0, -10.0])  # Lower param boundaries
  pmax     = np.array([ 40.0,  20.0,  10.0])  # Upper param boundaries
  prior    = np.array([  0.0,   0.0,   0.0])
  priorlow = np.array([  0.0,   0.0,   0.0])  # Flat priors
  priorup  = np.array([  0.0,   0.0,   0.0])
  # prior and priorup are irrelevant if priorlow == 0 (for a given parameter)

  chisq, bestp, bestmodel, lsfit = mc3.fit.modelfit(params, quad,
                               data, uncert, indparams=indparams,
                               stepsize=stepsize, pmin=pmin, pmax=pmax,
                               prior=prior, priorlow=priorlow, priorup=priorup)

