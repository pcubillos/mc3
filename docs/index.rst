.. MC3 documentation master file, created by
   sphinx-quickstart on Tue Dec 15 19:45:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br/>

MC3: Multi-Core Markov-Chain Monte Carlo
========================================

|Build Status|
|docs|
|PyPI|
|License|

.. raw:: html

    <embed>
    <span class="__dimensions_badge_embed__"
        data-doi="10.3847/1538-3881/153/1/3"
        data-style="small_circle"
        data-legend="always">
    </span>
    <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8">
    </script>
    </embed>

-------------------------------------------------------------------

:Author:        Patricio Cubillos and collaborators (see :ref:`team`)
:Contact:       `patricio.cubillos[at]oeaw.ac.at`_
:Organizations: `Space Research Institute (IWF) <http://iwf.oeaw.ac.at/>`_
:Web Site:      https://github.com/pcubillos/MCcubed
:Date:          |today|

-------------------------------------------------------------------

Features
========

``MC3`` is a powerful Bayesian-statistics tool that offers:

- Levenberg-Marquardt least-squares optimization.
- Markov-chain Monte Carlo (MCMC) posterior-distribution sampling following the:

  - Metropolis-Hastings algorithm with Gaussian proposal distribution,
  - Differential-Evolution MCMC (DEMC), or
  - DEMCzs (Snooker).

The following features are available when running ``MC3``:

- Execution from the Shell prompt or interactively through the Python interpreter.
- Single- or multiple-CPU parallel computing.
- Uniform non-informative, Jeffreys non-informative, or Gaussian-informative priors.
- Gelman-Rubin convergence test.
- Share the same value among multiple parameters.
- Fix the value of parameters to constant values.
- Correlated-noise estimation with the Time-averaging or the Wavelet-based Likelihood estimation methods.

.. note:: ``MC3`` works in both Python2.7 and Python3!

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
- Monika Lendl (IWF)
- Ryan Challener (UCF)
- Michael Himes (UCF)

Documentation
=============

.. toctree::
   :maxdepth: 3

   getstarted
   mctutorial
   fittutorial
   timeaveraging
   api
   contributing
   license


Be Kind
=======

Please cite this paper if you found ``MC3`` useful for your research:
  `Cubillos et al. (2017): On the Correlated-noise Analyses Applied to Exoplanet Light Curves <http://ui.adsabs.harvard.edu/abs/2017AJ....153....3C>`_, AJ, 153, 3.

We welcome your feedback or inquiries, please refer them to:

  Patricio Cubillos (`patricio.cubillos[at]oeaw.ac.at`_)

``MC3`` is open-source open-development software under the MIT
:ref:`license`. |br|
Thank you for using ``MC3``!



.. _patricio.cubillos[at]oeaw.ac.at: patricio.cubillos@oeaw.ac.at
.. _Cubillos et al. 2017\: On the Correlated Noise Analyses Applied to Exoplanet Light Curves: http://adsabs.harvard.edu/abs/2017AJ....153....3C


.. |Build Status| image:: https://travis-ci.com/pcubillos/mccubed.svg?branch=master
   :target: https://travis-ci.com/pcubillos/mccubed

.. |docs| image:: https://readthedocs.org/projects/mccubed/badge/?version=latest
    :target: https://mccubed.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


.. |PyPI| image:: https://img.shields.io/pypi/v/mccubed.svg
    :target:      https://pypi.org/project/mccubed/
    :alt: Latest Version

.. |License| image:: https://img.shields.io/github/license/pcubillos/mccubed.svg?color=blue
    :target: https://mccubed.readthedocs.io/en/latest/license.html
