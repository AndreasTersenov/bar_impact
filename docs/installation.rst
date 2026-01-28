Installation
============

Requirements
------------

BAR_IMPACT requires Python 3.8 or later and the following core dependencies:

- numpy >= 1.20
- healpy >= 1.15
- h5py >= 3.0
- tqdm >= 4.60
- matplotlib >= 3.3
- scipy >= 1.7

Basic Installation
------------------

Install in development mode:

.. code-block:: bash

   git clone https://github.com/AndreasTersenov/bar_impact.git
   cd bar_impact
   pip install -e .

Optional Dependencies
---------------------

BAR_IMPACT has several optional dependency groups:

**Inference** (for NPE):

.. code-block:: bash

   pip install -e ".[inference]"

This installs:

- jax >= 0.3
- jaxlib >= 0.3
- jaxili >= 0.1
- getdist >= 1.3

**Coverage Testing** (for TARP):

.. code-block:: bash

   pip install -e ".[coverage]"

This installs:

- tarp >= 0.1.3

**Development** (for testing and linting):

.. code-block:: bash

   pip install -e ".[dev]"

This installs:

- pytest >= 7.0
- pytest-cov >= 3.0
- black >= 22.0
- isort >= 5.10
- ruff >= 0.1.0
- mypy >= 0.950
- pre-commit >= 3.0

**Documentation** (for building docs):

.. code-block:: bash

   pip install -e ".[docs]"

This installs:

- sphinx >= 6.0
- sphinx-rtd-theme >= 1.2
- sphinx-autodoc-typehints >= 1.22
- myst-parser >= 1.0

**All Dependencies**:

.. code-block:: bash

   pip install -e ".[all]"

External Dependencies
---------------------

Some features require packages not available on PyPI:

**pycs (CosmoStat)**

Required for wavelet L1 norm and peak count calculations.
Install from the CosmoStat repository:

.. code-block:: bash

   pip install git+https://github.com/CosmoStat/pycs.git

**jaxili**

Required for NPE inference. Install following the jaxili documentation.

Verifying Installation
----------------------

After installation, verify that the package is correctly installed:

.. code-block:: python

   import bar_impact
   print(bar_impact.__version__)

   # Check core functionality
   from bar_impact.core import ConvergenceMap, SurveyMask
   from bar_impact.processing import PowerSpectrumProcessor
   print("Core modules loaded successfully!")

   # Check optional dependencies
   try:
       from bar_impact.utils import initialize_npe
       print("NPE workflow available")
   except ImportError:
       print("NPE workflow not available (install inference extras)")

Running Tests
-------------

To verify the installation with the test suite:

.. code-block:: bash

   pytest tests/ -v

Tests requiring optional dependencies will be automatically skipped
if those dependencies are not installed.
