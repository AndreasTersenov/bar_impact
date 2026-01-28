Quick Start Guide
=================

This guide will help you get started with BAR_IMPACT for analyzing
cosmological weak lensing maps.

Loading Convergence Maps
------------------------

BAR_IMPACT provides the ``ConvergenceMap`` class for working with HEALPix maps:

.. code-block:: python

   from bar_impact.core import ConvergenceMap

   # Load from HDF5 file
   kappa = ConvergenceMap.from_h5("path/to/map.h5", bin_number=1)

   # Access the data
   print(f"Map shape: {kappa.data.shape}")
   print(f"NSIDE: {kappa.nside}")

Creating Survey Masks
---------------------

Use ``SurveyMask`` to create and apply survey masks:

.. code-block:: python

   from bar_impact.core import SurveyMask

   # Create a disk mask with specific sky coverage
   mask = SurveyMask.create_disk_mask(
       nside=512,
       target_area_sqdeg=14000
   )

   # Or create an apodized mask for smoother edges
   mask = SurveyMask.create_apodized_disk_mask(
       nside=512,
       target_area_sqdeg=14000,
       apodization_deg=2.0
   )

Adding Shape Noise
------------------

Add realistic shape noise to your maps:

.. code-block:: python

   # Add shape noise with default parameters
   kappa_noisy = kappa.add_shape_noise(sigma_e=0.26)

   # With custom galaxy density
   kappa_noisy = kappa.add_shape_noise(
       sigma_e=0.26,
       galaxy_density=6.75,
       seed=42  # For reproducibility
   )

Computing Power Spectra
-----------------------

Use ``PowerSpectrumProcessor`` for power spectrum analysis:

.. code-block:: python

   from bar_impact.processing import PowerSpectrumProcessor

   # Create processor
   processor = PowerSpectrumProcessor(lmax=1024)

   # Compute auto power spectrum
   cls = processor.process_single(kappa_noisy.data)

   # Compute with ell values
   cls, ell = processor.process_single(kappa_noisy.data, return_ell=True)

   # Compute cross power spectrum
   cls_cross = processor.process_cross(map1.data, map2.data)

   # Compute all cross spectra for multiple bins
   maps = [bin1.data, bin2.data, bin3.data, bin4.data]
   cls_dict = processor.process_all_cross_spectra(maps)

Computing L1 Norms
------------------

L1 norms capture non-Gaussian information through wavelet decomposition:

.. code-block:: python

   from bar_impact.processing import L1NormProcessor

   # Create processor (requires pycs)
   processor = L1NormProcessor(nscales=5, nbins=40)

   # Compute L1 norms
   l1_norms = processor.process_single(kappa_noisy.data, mask=mask.data)

Computing Peak Counts
---------------------

Peak counts measure the distribution of local maxima:

.. code-block:: python

   from bar_impact.processing import PeakCountProcessor

   # Create processor (requires pycs)
   processor = PeakCountProcessor(nscales=5, nbins=31)

   # Compute peak counts
   peaks = processor.process_single(kappa_noisy.data)

BNT Transform
-------------

Apply the BNT transform to null cross-correlations between redshift bins:

.. code-block:: python

   from bar_impact.processing import apply_bnt_transform
   from bar_impact.constants import BNT_MATRIX_DEFAULT

   # Load maps for all 4 bins
   maps = [ConvergenceMap.from_h5(f, bin_number=i) for i, f in enumerate(files, 1)]

   # Stack map data
   map_data = np.stack([m.data for m in maps])

   # Apply BNT transform
   bnt_maps = apply_bnt_transform(map_data, BNT_MATRIX_DEFAULT)

Using Constants
---------------

BAR_IMPACT provides standard constants:

.. code-block:: python

   from bar_impact.constants import (
       DEFAULT_NSIDE,      # 512
       DEFAULT_LMAX,       # 1024
       DEFAULT_SIGMA_E,    # 0.26
       DEFAULT_GALAXY_DENSITY,  # 6.75
       BNT_MATRIX_DEFAULT,      # 4x4 BNT matrix
       COSMO_PARAM_NAMES,       # Parameter names
   )

Logging
-------

BAR_IMPACT includes a logging infrastructure:

.. code-block:: python

   from bar_impact.utils.logging import get_logger, set_log_level
   import logging

   # Get a logger
   logger = get_logger(__name__)
   logger.info("Processing started")

   # Change log level
   set_log_level(logging.DEBUG)

   # Or via environment variable
   # export BAR_IMPACT_LOG_LEVEL=DEBUG

Error Handling
--------------

Use custom exceptions for better error handling:

.. code-block:: python

   from bar_impact.exceptions import (
       ProcessingError,
       MaskError,
       ConfigurationError,
   )

   try:
       result = processor.process_single(data)
   except MaskError as e:
       print(f"Mask issue: {e}")
   except ProcessingError as e:
       print(f"Processing failed: {e}")

Next Steps
----------

- See :doc:`api/index` for the complete API reference
- Check :doc:`workflows/index` for detailed workflow guides
- Read the example scripts in the ``examples/`` directory
