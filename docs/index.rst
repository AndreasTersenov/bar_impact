BAR_IMPACT
==========

A Python package for analyzing the impact of baryonic physics on cosmological weak lensing maps.

.. image:: https://github.com/AndreasTersenov/bar_impact/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/AndreasTersenov/bar_impact/actions/workflows/ci.yml

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

Overview
--------

BAR_IMPACT provides tools for:

- **Summary Statistics**: L1 norms, angular power spectra, peak counts
- **Tomographic Analysis**: BNT transform for nulling redshift correlations
- **Simulation-Based Inference**: Neural Posterior Estimation (NPE) with JAX
- **Posterior Validation**: TARP coverage testing

Quick Example
-------------

.. code-block:: python

   from bar_impact.core import ConvergenceMap, SurveyMask
   from bar_impact.processing import PowerSpectrumProcessor

   # Load and process a convergence map
   kappa = ConvergenceMap.from_h5("simulation.h5", bin_number=1)
   kappa = kappa.add_shape_noise(sigma_e=0.26)

   # Apply survey mask
   mask = SurveyMask.create_disk_mask(nside=512, target_area_sqdeg=14000)
   kappa = kappa.apply_mask(mask)

   # Compute power spectrum
   processor = PowerSpectrumProcessor(lmax=1024)
   cls = processor.process_single(kappa.data)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index
   workflows/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
