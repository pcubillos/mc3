.. _getstarted:

Getting Started
===============

System Requirements
-------------------

``MC3`` is compatible with Python 2.7 and 3.6+, and has been `tested
<https://travis-ci.com/pcubillos/mccubed>`_ to work on Unix/Linux and
OS X machines, with the following software:

* Numpy (version 1.8.2+)
* Scipy (version 0.17.1+)
* Matplotlib (version 1.3.1+)

``MC3`` may work with previous versions of these software;
however, we do not guarantee nor provide support for that.


Install
-------

To install ``MC3``, just run the following command (if you use conda, see instructions below):

.. code-block:: shell

    pip install mc3

Or alternatively (for conda users and for developers), clone the repository to your local machine with the following terminal commands:

.. code-block:: shell

    # Clone the repository to your working directory:
    git clone https://github.com/pcubillos/mc3
    cd mc3
    python setup.py install


To see the ``MC3`` docstring run:

.. code-block:: python

    import mc3
    help(mc3.mcmc)

Example 1: Interactive Run
--------------------------

The following example shows a basic MCMC run from the Python
interpreter, for a quadratic-polynomial fit to a noisy dataset:

.. code-block:: python

    import numpy as np
    import mc3

    def quad(p, x):
        """
        Quadratic polynomial function.

        Parameters
            p: Polynomial constant, linear, and quadratic coefficients.
            x: Array of dependent variables where to evaluate the polynomial.
        Returns
            y: Polinomial evaluated at x:  y = p0 + p1*x + p2*x^2
        """
        y = p[0] + p[1]*x + p[2]*x**2.0
        return y

    # Create a noisy synthetic dataset:
    x = np.linspace(0, 10, 1000)
    p_true = [3, -2.4, 0.5]
    y = quad(p_true, x)
    uncert = np.sqrt(np.abs(y))
    data = y + np.random.normal(0, uncert)

    # Initial guess for fitting parameters:
    params = np.array([10.0, -2.0, 0.1])
    pstep  = np.array([0.03, 0.03, 0.05])

    # Run the MCMC:
    func = quad
    mc3_results = mc3.mcmc(data, uncert, func, params, indparams=[x],
        pstep=pstep, sampler='snooker', nsamples=1e5, burnin=1000, ncpu=7)


That's it!.  The code returns a dictionary with the MCMC results.
Among these, you can find the best-fitting values (``bestp``),
the lower and upper boundaries of the 68%-credible region (``CRlo``
and ``CRhi``, with respect to ``bestp``), the standard deviation of
the marginal posteriors (``stdp``), the posterior sample
(``posterior``), and the chain index for each posterior sample
(``Zchain``).


``MC3`` will also print out to screen a progress report every 10% of
the MCMC run, showing the time, number of times a parameter tried to
go beyond the boundaries, the current best-fitting values, and
lowest :math:`\chi^{2}`; for example:

.. code-block:: none

  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    Multi-core Markov-chain Monte Carlo (MC3).
    Version 2.4.0.
    Copyright (c) 2015-2019 Patricio Cubillos and collaborators.
    MC3 is open-source software under the MIT license (see LICENSE).
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  Yippee Ki Yay Monte Carlo!
  Start MCMC chains  (Sun Nov  4 16:20:40 2018)

  [:         ]  10.0% completed  (Sun Nov  4 16:20:42 2018)
  Out-of-bound Trials:
  [0 0 0]
  Best Parameters: (chisq=1024.2992)
  [ 3.0603825  -2.42108869  0.50075726]

  ...

  [::::::::::] 100.0% completed  (Sun Nov  4 16:20:47 2018)
  Out-of-bound Trials:
  [0 0 0]
  Best Parameters: (chisq=1024.2772)
  [ 3.0679888  -2.4229654   0.50064008]

  Fin, MCMC Summary:
  ------------------
    Total number of samples:            100002
    Number of parallel chains:               7
    Average iterations per chain:        14286
    Burned-in iterations per chain:       1000
    Thinning factor:                         1
    MCMC sample size (thinned, burned):  93002
    Acceptance rate:   26.76%

  Param name     Best fit   Lo HPD CR   Hi HPD CR        Mean    Std dev       S/N
  ----------- ----------------------------------- ---------------------- ---------
  Param 1      3.0577e+00 -1.2951e-01  1.1875e-01  3.0555e+00 1.2384e-01      24.7
  Param 2     -2.4055e+00 -6.7695e-02  7.5366e-02 -2.4033e+00 7.1281e-02      33.7
  Param 3      4.9933e-01 -8.9207e-03  8.5756e-03  4.9902e-01 8.7305e-03      57.2

    Best-parameter's chi-squared:     1024.2772
    Bayesian Information Criterion:   1045.0004
    Reduced chi-squared:                 1.0274
    Standard deviation of residuals:  2.78898


At the end of the MCMC run, ``MC3`` displays a summary of the MCMC
sample, best-fitting parameters, credible-region boundaries, posterior
mean and standard deviation, among other statistics.

.. note:: More information will be displayed, depending on the MCMC
          configuration (see :ref:`mctutorial`).


Additionally, the user has the option to generate several plots of the MCMC
sample: the best-fitting model and data curves, parameter traces, and
marginal and pair-wise posteriors (these plots can also be generated
automatically with the MCMC run by setting ``plots=True``).
The plots sub-package provides the plotting functions:

.. code-block:: python

   import mc3.plots as mp
   import mc3.utils as mu

   # Output dict contains entire sample (Z), need to remove burn-in:
   posterior, Zchain, Zmask = mu.burn(mc3_results)
   bestp = mc3_results['bestp']
   # Set parameter names:
   pnames = ["constant", "linear", "quadratic"]

   # Plot best-fitting model and binned data:
   mp.modelfit(data, uncert, x, y, savefile="quad_bestfit.png")

   # Plot trace plot:
   mp.trace(posterior, Zchain, pnames=pnames, savefile="quad_trace.png")

   # Plot pairwise posteriors:
   mp.pairwise(posterior, pnames=pnames, bestp=bestp, savefile="quad_pairwise.png")

   # Plot marginal posterior histograms (with 68% highest-posterior-density credible regions):
   mp.histogram(posterior, pnames=pnames, bestp=bestp, percentile=0.683,
       savefile="quad_hist.png")

.. image:: ./quad_bestfit.png
   :width: 75%

.. image:: ./quad_trace.png
   :width: 75%

.. image:: ./quad_pairwise.png
   :width: 75%

.. image:: ./quad_hist.png
   :width: 75%


.. note:: These plots can also be automatically generated along with the
          MCMC run (see :ref:`outputs`).

Example 2: Shell Run
--------------------

The following example shows a basic MCMC run from the terminal using a
configuration file.
First, create a Python file ('*quadratic.py*') with the modeling function:

.. code-block:: python

    def quad(p, x):
        y = p[0] + p[1]*x + p[2]*x**2.0
        return y

Then, generate a data set and store into files, e.g., with the
following Python script:

.. code-block:: python

    import numpy as np
    import mc3
    from quadratic import quad

    # Create synthetic dataset:
    x  = np.linspace(0, 10, 1000)         # Independent model variable
    p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
    y  = quad(p0, x)                      # Noiseless model
    uncert = np.sqrt(np.abs(y))           # Data points uncertainty
    error  = np.random.normal(0, uncert)  # Noise for the data
    data   = y + error                    # Noisy data set
    # Store data set and other inputs:
    mc3.utils.savebin([data, uncert], 'data.npz')
    mc3.utils.savebin([x],            'indp.npz')

Now, create a configuration file with the ``MC3`` setup ('*MCMC.cfg*'):

.. code-block:: shell

    [MCMC]
    data      = data.npz
    indparams = indp.npz

    func     = quad quadratic
    params   =  10.0   -2.0   0.1
    pmin     = -25.0  -10.0 -10.0
    pmax     =  30.0   10.0  10.0
    pstep    =   0.3    0.3   0.05

    nsamples = 1e5
    burnin   = 1000
    ncpu     = 7
    sampler  = snooker
    grtest   = True
    plots    = True
    savefile = output_demo.npz


Finally, call the ``MC3`` entry point, providing the configuration file as
a command-line argument:

.. code-block:: shell

   mc3 -c MCMC.cfg


Troubleshooting
---------------

There may be an error with the most recent version of the
``multiprocessing`` module (version 2.6.2.1).  If the MCMC breaks with
an "AttributeError: __exit__" error message pointing to a
``multiprocessing`` module, try installing a previous version of it with
this shell command:

.. code-block:: shell

   pip install --upgrade 'multiprocessing<2.6.2'


