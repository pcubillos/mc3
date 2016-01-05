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
:Organizations: University of Central Florida (UCF), `Space Research Institute (IWF)`_
:Web Site:     https://github.com/pcubillos/MCcubed
:Date:         |today|



Features
--------

MC3 is a powerful Bayesian-statistics tool that offers:

- Levenberg-Marquardt least-squares optimization.
- Markov-chain Monte Carlo (MCMC) posterior-distribution sampling following the:

  - Metropolis-Hastings algorithm with Gaussian proposal distribution, or
  - Differential-Evolution MCMC (recomended).

The following features are available when running MC3:

- Execution from the Shell prompt or interactively through the Python interpreter.
- Single- or multiple-CPU parallel computing.
- Uniform non-informative, Jeffreys non-informative, or Gaussian-informative priors.
- Gelman-Rubin convergence test.
- Share the same value among multiple parameters.
- Fix the value of parameters to constant values.
- Correlated-noise estimation with the Time-averaging or the Wavelet-based Likelihood estimation methods.

.. _team:

Team members
------------

- `Patricio Cubillos`_ (UCF, IWF) `patricio.cubillos[at]oeaw.ac.at`_
- Joseph Harrington (UCF)
- Nate Lust (UCF)
- `AJ Foster`_ (UCF)
- Madison Stemm (UCF)

License
-------

MC3 is open-source open-development software under the MIT :ref:`license`.

Be Kind
-------

Please cite this paper if you found MC3 useful for your research:
  `Cubillos et al. 2016: On the Correlated Noise Analyses Applied to Exoplanet Light Curves`_, in preparation.

We welcome your feedback, but do not necessarily guarantee support.
Please send feedback or inquiries to:

  Patricio Cubillos (`patricio.cubillos[at]oeaw.ac.at`_)

Thank you for using MC3!

Contents
========

.. toctree::
   :maxdepth: 3

   getstarted
   tutorial
   license

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


.. _Patricio Cubillos: https://github.com/pcubillos/
.. _patricio.cubillos[at]oeaw.ac.at: patricio.cubillos@oeaw.ac.at
.. _Space Research Institute (IWF): http://iwf.oeaw.ac.at/
.. _AJ Foster: http://aj-foster.com
.. _Cubillos et al. 2016\: On the Correlated Noise Analyses Applied to Exoplanet Light Curves: https://github.com/pcubillos/MCcubed/lalala
