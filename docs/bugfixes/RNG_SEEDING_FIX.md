# RNG Seeding Fix for Multiprocessing

## Problem Identified

Both `cross_power_spectrum_processing.py` and `l1_norm_processing_new.py` had a **critical RNG seeding issue** that could compromise the validity of the noise realizations:

### Issues:
1. **No worker initialization**: Multiprocessing pools were created without an initializer function
2. **Shared RNG state**: Worker processes could inherit the same random number generator state from the parent process
3. **Correlated noise**: This could lead to identical or correlated noise realizations across different workers
4. **Non-reproducible results**: Results could vary between runs in unpredictable ways

## Root Cause

```python
# BEFORE (PROBLEMATIC):
with mp.Pool(processes=args.num_workers) as pool:
    # Workers share RNG state from parent!
```

When multiprocessing workers fork from the parent process, they inherit the parent's RNG state. Without proper reseeding, multiple workers could generate **identical noise sequences**, especially if they start processing at similar times.

## Solution Implemented

Added proper RNG initialization for each worker process using OS entropy:

```python
def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    # Use a source of entropy from the OS to seed the worker
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
```

```python
# AFTER (FIXED):
with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
    # Each worker gets a unique random seed from OS entropy!
```

## Why This Works

1. **`os.urandom(4)`**: Gets 4 bytes of cryptographically secure random data from OS entropy sources (e.g., `/dev/urandom` on Linux)
2. **Unique per worker**: Each worker process calls `seed_worker()` independently and gets a different seed
3. **Independent noise**: Each worker generates statistically independent noise realizations
4. **Thread-safe**: OS entropy sources are designed for concurrent access

## Files Modified

### Core Processing Scripts:
1. ✅ **`scripts/cross_power_spectrum_processing.py`**
   - Added `seed_worker()` function (line ~17)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~445)

2. ✅ **`scripts/l1_norm_processing_new.py`**
   - Added `seed_worker()` function (line ~34)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~198)

### BNT-Transformed Processing Scripts:
3. ✅ **`scripts/bnt_cross_power_spectrum_processing.py`**
   - Added `seed_worker()` function (line ~17)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~400)

4. ✅ **`scripts/bnt_power_spectrum_processing.py`**
   - Added `seed_worker()` function (line ~22)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~192)

5. ✅ **`scripts/bnt_l1_norm_processing_new.py`**
   - Added `seed_worker()` function (line ~40)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~213)

6. ✅ **`scripts/bnt_l1_norm_processing.py`**
   - Added `seed_worker()` function (line ~39)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~283)

7. ✅ **`scripts/bnt_peak_counts_processing_new.py`**
   - Added `seed_worker()` function (line ~40)
   - Updated `mp.Pool()` to include `initializer=seed_worker` (line ~253)

## Verification

To verify the fix is working, you can add temporary logging:

```python
def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    print(f"Worker {os.getpid()} initialized with seed: {seed}")  # Temporary debug
```

You should see different seed values for each worker process.

## Impact

### Before Fix:
- ❌ Workers could have correlated noise realizations
- ❌ Scientific validity questionable
- ❌ Non-reproducible in a controlled way
- ❌ Potential bias in statistical analysis

### After Fix:
- ✅ Each worker generates independent noise
- ✅ Statistically valid noise realizations
- ✅ Proper variance in simulated data
- ✅ Scientifically sound results

## Reference Implementation

This fix follows the same pattern used in `scripts/power_spectrum_processing.py`, which already had proper RNG seeding implemented.

## Important Note

While each **worker** now has a unique seed, different **runs** of the script will still produce different results (as intended for Monte Carlo simulations). If you need exact reproducibility across runs, you would need to:

1. Set a global seed before creating the pool: `np.random.seed(42)`
2. Use worker-specific but deterministic seeds: `np.random.seed(worker_id + base_seed)`

However, for cosmological simulations where you want realistic noise variance, the current implementation with OS entropy is **correct and preferred**.
