BAR_IMPACT Documentation
========================

**BAR_IMPACT** is a Python package for analyzing the impact of baryonic physics
on cosmological weak lensing maps. It provides tools for wavelet analysis,
power spectra computation, peak counting, and Neural Posterior Estimation (NPE)
for simulation-based inference.

.. note::

   This project is under active development.

Features
--------

- **Convergence Map Processing**: Load, manipulate, and process HEALPix convergence maps
- **Multiple Summary Statistics**: L1 norms, power spectra, peak counts
- **BNT Transform**: Bernardeau-Nishimichi-Taruya transform for tomographic analysis
- **NPE Inference**: Neural Posterior Estimation using JAX and jaxili
- **Coverage Testing**: TARP-based coverage diagnostics
- **Reproducibility**: Deterministic seeding for consistent results

Quick Start
-----------

.. code-block:: python

   from bar_impact.core import ConvergenceMap, SurveyMask
   from bar_impact.processing import PowerSpectrumProcessor
   from bar_impact.constants import BNT_MATRIX_DEFAULT

   # Load a convergence map
   kappa = ConvergenceMap.from_h5("map.h5", bin_number=1)

   # Add shape noise and apply mask
   mask = SurveyMask.create_disk_mask(nside=512, target_area_sqdeg=14000)
   kappa_noisy = kappa.add_shape_noise(sigma_e=0.26)
   kappa_masked = kappa_noisy.apply_mask(mask)

   # Compute power spectrum
   processor = PowerSpectrumProcessor(lmax=1024)
   cls = processor.process_single(kappa_masked.data)

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install -e .

   # With all optional dependencies
   pip install -e ".[all]"

   # With specific extras
   pip install -e ".[inference]"  # JAX, jaxili for NPE
   pip install -e ".[dev]"        # Development tools

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Workflows

   workflows/index

Development
-----------

- `CONTRIBUTING.md <https://github.com/AndreasTersenov/bar_impact/blob/main/CONTRIBUTING.md>`_ - Development guidelines
- `CHANGELOG.md <https://github.com/AndreasTersenov/bar_impact/blob/main/CHANGELOG.md>`_ - Version history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
