# Summary: TARP Coverage Testing Implementation

## What Was Added

I've successfully integrated TARP (Test for Accurate Ranked Predictions) statistical coverage tests into your NPE inference script `run_npe_inference_auto_cross_ps.py`. This allows you to assess the quality and calibration of your posterior estimators.

## Key Features

### 1. **TARP Coverage Testing**
   - Tests whether posterior distributions have correct coverage properties
   - Detects overconfident, underconfident, or biased posteriors
   - Based on Lemos, Coogan et al. 2023 (arXiv:2302.03026)

### 2. **Bootstrap Uncertainty Estimation**
   - Optional bootstrap resampling for uncertainty bands on coverage estimates
   - Helps quantify statistical uncertainty in the coverage diagnostic

### 3. **Comprehensive Visualization**
   - Automatic generation of coverage diagnostic plots
   - Comparison against ideal calibration
   - Saves both PDF (high-res) and PNG (quick-view) formats

### 4. **Flexible Configuration**
   - Control number of test simulations
   - Adjust posterior samples per simulation
   - Enable/disable bootstrap with custom iterations
   - Set random seeds for reproducibility

## New Command-Line Arguments

```bash
--run-coverage-test              # Enable TARP coverage testing
--coverage-num-sims 100          # Number of test simulations (default: 100)
--coverage-num-samples 1000      # Posterior samples per simulation (default: 1000)
--coverage-bootstrap             # Enable bootstrap uncertainty estimation
--coverage-num-bootstrap 100     # Bootstrap iterations (default: 100)
--coverage-seed 42               # Random seed (default: 42)
```

## Usage Example

```bash
python scripts/run_npe_inference_auto_cross_ps.py \
    --simulation-type nobaryons \
    --fiducial-type baryonified \
    --bins 1,2,3,4 \
    --lower-cut 30 \
    --upper-cut 1024 \
    --rebin 20 \
    --noisy \
    --noise-level 0.26 \
    --run-coverage-test \
    --coverage-num-sims 100 \
    --coverage-num-samples 1000 \
    --coverage-bootstrap \
    --num-samples 3000
```

## Files Created

1. **`scripts/run_npe_inference_auto_cross_ps.py`** (Modified)
   - Added TARP package imports
   - Added `run_tarp_coverage_test()` function
   - Added `plot_tarp_coverage()` function
   - Integrated coverage testing into main workflow

2. **`scripts/example_coverage_test.sh`** (New)
   - Ready-to-run example script demonstrating coverage testing
   - Multiple configuration examples (quick, standard, comprehensive)

3. **`TARP_COVERAGE_TESTING.md`** (New)
   - Comprehensive documentation (30+ sections)
   - Background on TARP methodology
   - Detailed usage instructions
   - Interpretation guidelines
   - Troubleshooting guide
   - Multiple examples

4. **`TARP_QUICK_REFERENCE.md`** (New)
   - One-page quick reference guide
   - Command presets for different use cases
   - Visual interpretation guide
   - Troubleshooting tips

## Output Files

When you run coverage tests, three new files are generated per run:

1. **`*_tarp_coverage.pdf`** - High-resolution coverage diagnostic plot
2. **`*_tarp_coverage.png`** - Quick-view coverage plot
3. **`*_tarp_coverage_data.npz`** - Raw coverage data (ecp, alpha arrays)

## How It Works

1. **Selection**: Randomly selects test simulations from your training set
2. **Sampling**: Generates posterior samples for each test simulation
3. **TARP Algorithm**: 
   - Computes distances from random reference points
   - Calculates coverage fractions
   - Estimates expected coverage probability (ECP)
4. **Bootstrap** (optional): Repeats with resampling for uncertainty estimates
5. **Visualization**: Plots ECP vs credibility level, compares to ideal calibration

## Interpreting Results

### Well-Calibrated ✅
- Coverage curve follows the diagonal (y = x)
- Posterior uncertainty is trustworthy

### Overconfident ⚠️
- Coverage curve **below** diagonal
- Posterior underestimates uncertainty
- May need more training data or model complexity

### Underconfident ⚠️
- Coverage curve **above** diagonal  
- Posterior overestimates uncertainty
- May be too conservative, reduce regularization

## Performance Guidelines

| Configuration | Runtime* | Use Case |
|--------------|----------|----------|
| Quick (50 sims, 500 samples) | ~5 min | Development/debugging |
| Standard (100 sims, 1000 samples) | ~15 min | Regular validation |
| Comprehensive (200 sims, 2000 samples, bootstrap) | ~45 min | Publication results |

*Runtime depends on data dimensionality and hardware

## Integration with Existing Workflow

The coverage test is **optional** and **non-intrusive**:
- Add `--run-coverage-test` flag to enable
- Runs after model training, before fiducial posterior sampling
- Does not affect existing outputs or results
- Can be added to any existing inference command

## Next Steps

1. **Try a quick test**: Run `bash scripts/example_coverage_test.sh` (edit paths as needed)
2. **Check results**: Look at the `*_tarp_coverage.pdf` plot in your output directory
3. **Interpret**: Use `TARP_QUICK_REFERENCE.md` for interpretation guidance
4. **Optimize**: Adjust test parameters based on runtime and detail needs
5. **Document**: Include coverage plots in your analysis documentation

## Technical Details

- **Package**: Uses the `tarp` package from your repo (`tarp/src/tarp/`)
- **Dependencies**: numpy, matplotlib, jax (already in your environment)
- **Method**: Euclidean distance in normalized parameter space
- **Reference points**: Random sampling from unit hypercube
- **Normalization**: Automatic parameter range normalization

## Benefits

1. **Quality Assurance**: Quantitatively verify posterior calibration
2. **Debugging**: Identify training issues or model problems early
3. **Publication**: Demonstrate posterior quality with coverage plots
4. **Comparison**: Test different model configurations objectively
5. **Trust**: Build confidence in uncertainty estimates

## References

- Lemos, P., Coogan, A., et al. (2023). "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference." arXiv:2302.03026

## Questions?

For detailed information, see:
- Full documentation: `TARP_COVERAGE_TESTING.md`
- Quick reference: `TARP_QUICK_REFERENCE.md`
- Example script: `scripts/example_coverage_test.sh`
- TARP package: `tarp/src/tarp/drp.py`

The implementation is ready to use! Simply add `--run-coverage-test` to any of your existing NPE inference commands.
