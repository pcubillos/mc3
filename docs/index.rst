.. MC3 documentation master file, created by
   sphinx-quickstart on Tue Dec 15 19:45:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br/>

Multi-Core Markov-Chain Monte Carlo (MC3)
=========================================

:Author:       Patricio Cubillos and collaborators (see :ref:`team`)
:Contact:      `patricio.cubillos[at]oeaw.ac.at`_
:Organizations: University of Central Florida (UCF), `Space Research Institute (IWF) <http://iwf.oeaw.ac.at/>`_
:Web Site:     https://github.com/pcubillos/MCcubed
:Date:         |today|



Features
========

``MC3`` is a powerful Bayesian-statistics tool that offers:

- Levenberg-Marquardt least-squares optimization.
- Markov-chain Monte Carlo (MCMC) posterior-distribution sampling following the:

  - Metropolis-Hastings algorithm with Gaussian proposal distribution,
  - Differential-Evolution MCMC (DEMC), or
  - DEMCzs (Snooker) (Recommended).

The following features are available when running ``MC3``:

- Execution from the Shell prompt or interactively through the Python interpreter.
- Single- or multiple-CPU parallel computing.
- Uniform non-informative, Jeffreys non-informative, or Gaussian-informative priors.
- Gelman-Rubin convergence test.
- Share the same value among multiple parameters.
- Fix the value of parameters to constant values.
- Correlated-noise estimation with the Time-averaging or the Wavelet-based Likelihood estimation methods.

.. note:: ``MC3`` should work for both Python2.7 and Python3!

.. _team:

Collaborators
=============

All of these people have made a direct or indirect contribution to
``MCcubed``, and in many instances have been fundamental in the
development of this package.

- `Patricio Cubillos <https://github.com/pcubillos>`_ (UCF, IWF) `patricio.cubillos[at]oeaw.ac.at`_
- Joseph Harrington (UCF)
- Nate Lust (UCF)
- `AJ Foster <http://aj-foster.com>`_ (UCF)
- Madison Stemm (UCF)
- Tom Loredo (Cornell)
- Kevin Stevenson (UCF)
- Chris Campo (UCF)
- Matt Hardin (UCF)
- Ryan Hardy (UCF)

Documentation
=============

.. toctree::
   :maxdepth: 3

   getstarted
   mctutorial
   fittutorial
   timeaveraging
   contributing
   license

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


Be Kind
=======


Please cite this paper if you found ``MC3`` useful for your research:
  `Cubillos et al. 2017: On the Correlated-noise Analyses Applied to
  Exoplanet Light Curves <http://adsabs.harvard.edu/abs/2017AJ....153....3C>`_, AJ, 153, 3.

We welcome your feedback, but do not necessarily guarantee support.
Please send feedback or inquiries to:

  Patricio Cubillos (`patricio.cubillos[at]oeaw.ac.at`_)

``MC3`` is open-source open-development software under the MIT
:ref:`license`.

Thank you for using ``MC3``!

Documentation for Previous Releases
===================================

- `MC3 version 1.1 <http://geco.oeaw.ac.at/patricio/MC3_v1.1.pdf>`_.
- `MC3 version 1.2 <http://geco.oeaw.ac.at/patricio/MC3_v1.2.pdf>`_.

.. _patricio.cubillos[at]oeaw.ac.at: patricio.cubillos@oeaw.ac.at
.. _Cubillos et al. 2017\: On the Correlated Noise Analyses Applied to Exoplanet Light Curves: http://adsabs.harvard.edu/abs/2017AJ....153....3C
