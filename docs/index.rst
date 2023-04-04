.. MC3 documentation master file, created by
   sphinx-quickstart on Tue Dec 15 19:45:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br/>

mc3: Multi-Core Markov-Chain Monte Carlo
========================================

|Build Status|
|docs|
|PyPI|
|conda|
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
:Web Site:      https://github.com/pcubillos/mc3
:Date:          |today|

-------------------------------------------------------------------

.. note::

    Got  Python3.6+? Simply install as: ``pip install mc3``


Features
========

``mc3`` is a Bayesian-statistics tool that offers:

- Levenberg-Marquardt least-squares optimization.
- Markov-chain Monte Carlo (MCMC) posterior-distribution sampling following the:

  - Metropolis-Hastings algorithm with Gaussian proposal distribution,
  - Differential-Evolution MCMC (DEMC), or
  - DEMCzs (Snooker).

.. - Nested-sampling via `dynesty <https://dynesty.readthedocs.io/en/latest/>`_.

The following features are available when running ``mc3``:

- Execution from the Shell prompt or interactively through the Python interpreter.
- Single- or multiple-CPU parallel computing.
- Uniform non-informative, Jeffreys non-informative, or Gaussian-informative priors.
- Gelman-Rubin convergence test.
- Share the same value among multiple parameters.
- Fix the value of parameters to constant values.
- Correlated-noise estimation with the Time-averaging or the Wavelet-based Likelihood estimation methods.

.. _team:

Collaborators
=============

All of these people have made a direct or indirect contribution to
``mc3``, and in many instances have been fundamental in the
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

   get_started
   mcmc_tutorial
   fit_tutorial
   time_averaging
   references
   api
   contributing
   license

..   ns_tutorial

Be Kind
=======

Please cite this paper if you found ``mc3`` useful for your research:
  `Cubillos et al. (2017): On the Correlated-noise Analyses Applied to Exoplanet Light Curves <http://ui.adsabs.harvard.edu/abs/2017AJ....153....3C>`_, AJ, 153, 3.

We welcome your feedback or inquiries, please refer them to:

  Patricio Cubillos (`patricio.cubillos[at]oeaw.ac.at`_)

``mc3`` is open-source open-development software under the MIT
:ref:`license`. |br|
Thank you for using ``mc3``!



.. _patricio.cubillos[at]oeaw.ac.at: patricio.cubillos@oeaw.ac.at
.. _Cubillos et al. 2017\: On the Correlated Noise Analyses Applied to Exoplanet Light Curves: http://adsabs.harvard.edu/abs/2017AJ....153....3C


.. |Build Status| image:: https://github.com/pcubillos/mc3/actions/workflows/python-package.yml/badge.svg?branch=master
    :target: https://github.com/pcubillos/mc3/actions/workflows/python-package.yml?query=branch%3Amaster


.. |docs| image:: https://readthedocs.org/projects/mc3/badge/?version=latest
    :target: https://mc3.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


.. |PyPI| image:: https://img.shields.io/pypi/v/mc3.svg
    :target:      https://pypi.org/project/mc3/
    :alt: Latest Version

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/mc3.svg
    :target: https://anaconda.org/conda-forge/mc3


.. |License| image:: https://img.shields.io/github/license/pcubillos/mc3.svg?color=blue
    :target: https://mc3.readthedocs.io/en/latest/license.html

