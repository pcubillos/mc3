.. _nstutorial:


Nested Sampling Tutorial
========================

This tutorial describes the available options when running Nested
Sampling with ``MC3``.  The following sections make up a script meant
to be run from the Python interpreter or in a Python script.  At the
:ref:`bottom of this page <ns-run>` you can see the entire script.

Preamble
--------

``MC3`` implements Nested Sampling through the `dynesty
<https://dynesty.readthedocs.io/en/latest/index.html>`_ package
[Speagle2019]_.  Thus, make sure to install and cite this Python
package if needed.

In this tutorial, we will use the same :ref:`preamble` setup as in
the MCMC tutorial, fitting a quadratic polynomial.  Likewise, most of
the input arguments follow the same format as an MCMC run, including:
:ref:`data`, :ref:`func`, :ref:`priors`, :ref:`pnames`,
:ref:`optimization`, :ref:`outputs`, and :ref:`logging`.


Sampler Algorithm
-----------------

Set the ``sampler`` argument to ``dynesty`` for a nested-sampling run
with dynesty:

.. literalinclude:: ../examples/ns_tutorial.py
    :lines: 52-53


Fitting Parameters
------------------

The ``params`` argument (required) is a 1D float ndarray containing
the initial-guess values for the model fitting parameters.

A nested-sampling run requires a proper domain (i.e., bounded); thus,
the ``pmin`` and ``pmax`` arguments are required, and must have finite
values.

The ``pstep`` argument sets the sampling behavior of the fitting
parameters.  A positive ``pstep`` value leaves a parameter free, a
``pstep`` value of zero keeps the parameter fixed, whereas a negative
``pstep`` value make the parameter to share its value with another
parameter (see :ref:`behavior`).

The ``prior``, ``priorlow``, and ``priorup`` arguments set the type of
prior (uniform or Gaussian) and their values.  See details in
:ref:`priors`.

.. literalinclude:: ../examples/ns_tutorial.py
    :lines: 35-46


Sampling Configuration
----------------------

The following arguments set the nested- configuration:

.. literalinclude:: ../examples/ns_tutorial.py
    :lines: 55-60

The ``leastsq`` argument (optional, default: None) allows to run a
least-squares optimization before the sampling (see
:ref:`optimization` for details).

The ``ncpu`` argument (optional, default: ``nchains``) sets the number
CPUs to use for the sampling.  When ``ncpu>1``, ``MC3`` will run in
parallel processors through the ``mutiprocessing`` Python
Standard-Library package (no need to set a ``pool`` input).

The ``thinning`` argument (optional, default: 1) sets the posterior
thinning factor (discarding all but every ``thinning``-th sample),
to reduce the memory usage.


Furhter dynesty Configuration
-----------------------------

An ``mc3.sample()`` run with dynesty nested-sampling can also
receive arguments accepted by `dynesty.DynamicNestedSampler()
<https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynesty.DynamicNestedSampler>`_
or `run_nested()
<https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.DynamicSampler.run_nested>`_
method.

However, note that if you pass ``prior_transform``, ``MC3`` won't be
able to compute the log(posterior) (implementation is TBD).  Likewise,
if you pass ``loglikelihood`` or ``prior_transform``, ``MC3`` won't be
able to run an optimization (implementation is TBD).


.. _ns-run:

Nested-sampling Run
-------------------

Putting it all together, here's a Python script to run an ``MC3``
nested-sampling retrieval:

.. literalinclude:: ../examples/ns_tutorial.py

A nested-sampling run returns a dictionary with the same outputs as an
MCMC run (see :ref:`outputs`), except that instead of an
``acceptance_rate``, it contains the sampling efficiency ``eff``, and
the dynesty sampler object ``dynesty_sampler``.  The screen output
should look like this:

.. code-block:: none

    ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
      Multi-core Markov-chain Monte Carlo (MC3).
      Version 3.0.0.
      Copyright (c) 2015-2019 Patricio Cubillos and collaborators.
      MC3 is open-source software under the MIT license (see LICENSE).
    ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Least-squares best-fitting parameters:
      [ 3.47315859 -2.71597145  0.53126294]

    Running dynesty dynamic nested-samping run:
    iter: 23417|eff(%): 26.047|logl*:  -63.0< -56.3< -57.3|logz:  -69.6+/-0.2

    Nested Sampling Summary:
    ------------------------
      Number of evaluated samples:  23417
      Thinning factor:                  1
      NS sample size (thinned):     23417
      Sampling efficiency:  26.05%

    Param name     Best fit   Lo HPD CR   Hi HPD CR        Mean    Std dev       S/N
    ----------- ----------------------------------- ---------------------- ---------
    y0           3.4732e+00 -1.1103e-01  1.6907e-01  3.5370e+00 1.4736e-01      23.6
    alpha       -2.7160e+00 -1.2635e-01  8.2435e-02 -2.7514e+00 1.0505e-01      25.9
    beta         5.3126e-01 -1.4149e-02  2.0976e-02  5.3508e-01 1.7523e-02      30.3

      Best-parameter's chi-squared:       113.6593
      Best-parameter's -2*log(posterior): 113.7313
      Bayesian Information Criterion:     127.4748
      Reduced chi-squared:                  1.1717
      Standard deviation of residuals:  3.0212

    Output sampler files:
      'NS_tutorial.npz'
      'NS_tutorial_trace.png'
      'NS_tutorial_pairwise.png'
      'NS_tutorial_posterior.png'
      'NS_tutorial_RMS.png'
      'NS_tutorial_model.png'
      'NS_tutorial.log'

