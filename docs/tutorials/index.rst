Tutorials
=========

This section contains tutorials for common tasks with BAR_IMPACT.

Power Spectrum Analysis
-----------------------

Learn how to compute and analyze power spectra from convergence maps.

.. code-block:: python

   from bar_impact.core import ConvergenceMap, SurveyMask
   from bar_impact.processing import PowerSpectrumProcessor

   # Load map and create mask
   kappa = ConvergenceMap.from_h5("map.h5", bin_number=1)
   mask = SurveyMask.create_apodized_disk_mask(nside=512, target_area_sqdeg=14000)

   # Add noise and apply mask
   kappa_noisy = kappa.add_shape_noise(sigma_e=0.26, seed=42)
   kappa_masked = kappa_noisy.apply_mask(mask)

   # Compute power spectrum
   processor = PowerSpectrumProcessor(lmax=1024, ell_min=100, ell_max=900)
   cls, ell = processor.process_single(kappa_masked.data, return_ell=True)

L1 Norm Analysis
----------------

Computing wavelet L1 norms for non-Gaussian statistics.

.. code-block:: python

   from bar_impact.processing import L1NormProcessor

   # Create processor
   processor = L1NormProcessor(nscales=5, nbins=40)

   # Important: Pass mask to processor, don't pre-multiply
   l1_norms = processor.process_single(kappa_noisy.data, mask=mask.data)

BNT Transform Workflow
----------------------

Applying the BNT transform for tomographic analysis.

.. code-block:: python

   import numpy as np
   from bar_impact.processing import apply_bnt_transform
   from bar_impact.constants import BNT_MATRIX_DEFAULT

   # Stack 4 redshift bin maps
   maps_stacked = np.stack([bin1.data, bin2.data, bin3.data, bin4.data])

   # Apply BNT transform
   bnt_maps = apply_bnt_transform(maps_stacked, BNT_MATRIX_DEFAULT)

   # Now process the BNT-transformed maps
   # Note: Apply mask BEFORE BNT transform for proper handling

NPE Inference
-------------

Running Neural Posterior Estimation (requires jaxili).

.. code-block:: python

   from bar_impact.inference import NPEInference, NPEConfig

   # Configure NPE
   config = NPEConfig(
       num_epochs=1000,
       learning_rate=1e-4,
       batch_size=40,
   )

   # Create inference object
   npe = NPEInference(config=config)

   # Train on simulations
   npe.train(data_vectors, parameters)

   # Sample posterior for observed data
   result = npe.sample(observed_data, num_samples=10000)

   # Get statistics
   print(result.summary())

See Also
--------

- :doc:`../workflows/index` for detailed workflow documentation
- :doc:`../api/index` for complete API reference
