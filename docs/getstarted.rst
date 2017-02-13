.. _getstarted:

Getting Started
===============

System Requirements
-------------------

``MC3`` (version 2.2) is known to work on Unix/Linux (Ubuntu)
and OSX (10.9+) machines, with the following software:

* Python (version 2.7+ or 3.4+)
* Numpy (version 1.8.2+)
* Scipy (version 0.17.1+)
* Matplotlib (version 1.3.1+)

``MC3`` may work with previous versions of these software;
however, we do not guarantee nor provide support for that.

Install
-------

To obtain the latest MCcubed code, clone the repository to your local
machine with the following terminal commands.
First, keep track of the folder where you are putting ``MC3``:

.. code-block:: shell

  topdir=`pwd`
  git clone https://github.com/pcubillos/MCcubed

Compile
-------

To compile the C-extensions of the package run:


.. code-block:: shell

  cd $topdir/MCcubed/
  make

To compile the documentation of the package, run:

.. code-block:: shell

  cd $topdir/MCcubed/docs
  make latexpdf

A pdf version of this documentation will be available at
``$topdir/MCcubed/docs/latex/MC3.pdf``.  To remove the program
binaries, run:

.. code-block:: shell

  cd $topdir/MCcubed/
  make clean

..  Documentation
    -------------

  To see the MCMC docstring run:

  .. code-block:: python

     import mccubed as mc3
     help(mc3.mcmc)

Example 1 Interactive
---------------------

The following example (`demo01 <https://github.com/pcubillos/MCcubed/blob/master/examples/demo01/demo01.py>`_) shows a basic MCMC run with ``MC3`` from
the Python interpreter.
This example fits a quadratic polynomial curve to a dataset.
First create a folder to run the example (alternatively, run the example
from any location, but adjust the paths of the Python script):

.. code-block:: shell

   cd $topdir
   mkdir run01
   cd run01

Now start a Python interactive session.  This script imports the
necesary modules, creates a noisy dataset, and runs the MCMC:

.. code-block:: python

   import sys
   import numpy as np

   sys.path.append("../MCcubed/")
   import MCcubed as mc3

   # Get function to model (and sample):
   sys.path.append("../MCcubed/examples/models/")
   from quadratic import quad

   # Create a synthetic dataset:
   x = np.linspace(0, 10, 1000)         # Independent model variable
   p0 = [3, -2.4, 0.5]                  # True-underlying model parameters
   y = quad(p0, x)                      # Noiseless model
   uncert = np.sqrt(np.abs(y))          # Data points uncertainty
   error = np.random.normal(0, uncert)  # Noise for the data
   data = y + error                     # Noisy data set

   # Fit the quad polynomial coefficients:
   params = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
   stepsize = np.array([0.03, 0.03, 0.05])

   # Run the MCMC:
   bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(data, uncert,
       func=quad, indparams=[x], params=params, stepsize=stepsize,
       nsamples=1e5, burnin=1000)

The code will return the best-fitting values (``bestp``), the lower
and upper boundaries of the 68%-credible region (``CRlo`` and
``CRhi``, with respect to ``bestp``), the standard deviation of the
marginal posteriors (``stdp``), the posterior sample (``posterior``),
and the chain index for each posterior sample (``Zchain``).

Outputs
^^^^^^^

That's it, now let's see the results.  ``MC3`` will print out to screen a
progress report every 10% of the MCMC run, showing the time, number of
times a parameter tried to go beyond the boundaries, the current
best-fitting values, and corresponding :math:`\chi^{2}`; for example:

.. code-block:: none

  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    Multi-Core Markov-Chain Monte Carlo (MC3).
    Version 2.2.11.
    Copyright (c) 2015-2016 Patricio Cubillos and collaborators.
    MC3 is open-source software under the MIT license (see LICENSE).
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  Start MCMC chains  (Fri Sep  2 12:12:55 2016)

  [:         ]  10.0% completed  (Fri Sep  2 12:12:55 2016)
  Out-of-bound Trials:
  [0 0 0]
  Best Parameters: (chisq=958.6322)
  [ 3.17360501 -2.49573272  0.51256399]

  ...

  [::::::::::] 100.0% completed  (Fri Sep  2 12:12:57 2016)
  Out-of-bound Trials:
  [0 0 0]
  Best Parameters: (chisq=958.6192)
  [ 3.15477168 -2.4840968   0.511011  ]

  Fin, MCMC Summary:
  ------------------
    Total number of samples:            100002
    Number of parallel chains:               7
    Average iterations per chain:        14286
    Burned in iterations per chain:       1000
    Thinning factor:                         1
    MCMC sample (thinned, burned) size:  93002
    Acceptance rate:   27.29%

        Best fit  Lo Cred.Reg.  Hi Cred.Reg.          Mean     Std. dev.      S/N
    3.154772e+00 -1.148446e-01  1.208576e-01  3.158164e+00  1.192656e-01     26.5
   -2.484097e+00 -6.988181e-02  6.490437e-02 -2.487244e+00  6.816345e-02     36.4
    5.110110e-01 -7.921301e-03  8.774663e-03  5.115275e-01  8.345855e-03     61.2

    Best-parameter's chi-squared:     958.6192
    Bayesian Information Criterion:   979.3424
    Reduced chi-squared:                0.9615
    Standard deviation of residuals:  2.65388


At the end of the MCMC run, ``MC3`` displays a summary of the MCMC
sample, best-fitting parameters, credible-region boundaries, posterior
mean and standard deviation, among other statistics.

.. note:: More information will be displayed, depending on the MCMC
          configuration (see the :ref:`mctutorial`).


Additionally, the user has the option to generate several plots of the MCMC
sample: the best-fitting model and data curves, parameter traces, and
marginal and pair-wise posteriors (these plots can also be generated
automatically with the MCMC run by setting ``plots=True``).
The plots sub-package provides the plotting functions:

.. code-block:: python

   # Plot best-fitting model and binned data:
   mc3.plots.modelfit(data, uncert, x, y, savefile="quad_bestfit.png")
   # Plot trace plot:
   parname = ["constant", "linear", "quadratic"]
   mc3.plots.trace(posterior, Zchain, parname=parname, savefile="quad_trace.png")

   # Plot pairwise posteriors:
   mc3.plots.pairwise(posterior, parname=parname, savefile="quad_pairwise.png")

   # Plot marginal posterior histograms (with 68% highest-posterior-density credible regions):
   mc3.plots.histogram(posterior, parname=parname, savefile="quad_hist.png",
                       percentile=0.683)

.. image:: ./quad_bestfit.png
   :width: 50%

.. image:: ./quad_trace.png
   :width: 50%

.. image:: ./quad_pairwise.png
   :width: 50%

.. image:: ./quad_hist.png
   :width: 50%


.. note:: These plots can also be automatically generated along with the
          MCMC run (see `File Outputs
          <http://pcubillos.github.io/MCcubed/tutorial.html#file-outputs>`_).

Example 2: Shell Run
--------------------

The following example
(`demo02 <https://github.com/pcubillos/MCcubed/blob/master/examples/demo02/>`_)
shows a basic MCMC run from the shell prompt.
To start, create a working directory to place the files and execute the program:

.. code-block:: shell

   cd $topdir
   mkdir run02
   cd run02


Copy the demo files (configuration and data files) to the run folder:

.. code-block:: shell

   cp $topdir/MCcubed/examples/demo02/* .


Call the ``MC3`` executable, providing the configuration file as
command-line argument:

.. code-block:: shell

   $topdir/MCcubed/mc3.py -c MCMC.cfg

Troubleshooting
---------------

There may be an error with the most recent version of the
``multiprocessing`` module (version 2.6.2.1).  If the MCMC breaks with
an "AttributeError: __exit__" error message pointing to a
``multiprocessing`` module, try installing a previous version of it with
this shell command:

.. code-block:: shell

   pip install --upgrade 'multiprocessing<2.6.2'


