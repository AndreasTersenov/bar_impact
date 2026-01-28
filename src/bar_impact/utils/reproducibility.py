"""
Reproducibility utilities for deterministic random number generation.

This module provides utilities for ensuring reproducible results across
different runs, including deterministic seed generation and worker initialization.
"""

import hashlib
import multiprocessing as mp
import os
from typing import Optional

import numpy as np

__all__ = [
    "get_deterministic_seed",
    "seed_worker",
    "create_seed_worker_initializer",
]


def get_deterministic_seed(file_path: str, global_seed: int = 42) -> int:
    """
    Generate a deterministic seed from file path and global seed.

    This ensures reproducibility across runs while giving each file
    a unique seed based on its path.

    Parameters
    ----------
    file_path : str
        Path to the file being processed
    global_seed : int, optional
        Global random seed (default: 42)

    Returns
    -------
    int
        Deterministic seed value in range [0, 2^32)

    Examples
    --------
    >>> seed1 = get_deterministic_seed("/path/to/file1.h5", 42)
    >>> seed2 = get_deterministic_seed("/path/to/file1.h5", 42)
    >>> seed1 == seed2
    True
    >>> seed3 = get_deterministic_seed("/path/to/file2.h5", 42)
    >>> seed1 != seed3
    True
    """
    hash_input = f"{file_path}_{global_seed}".encode("utf-8")
    hash_digest = hashlib.sha256(hash_input).digest()
    seed = int.from_bytes(hash_digest[:4], byteorder="big")
    return seed % (2**32)


def seed_worker(global_seed: Optional[int] = None):
    """
    Initialize worker process with a unique but deterministic seed.

    This function is used as an initializer for multiprocessing pools.
    Each worker gets a unique seed based on the global seed and worker ID.

    Parameters
    ----------
    global_seed : int, optional
        Global random seed. If None, uses OS entropy (non-deterministic).

    Examples
    --------
    >>> from multiprocessing import Pool
    >>> from functools import partial
    >>>
    >>> # Deterministic
    >>> with Pool(4, initializer=seed_worker, initargs=(42,)) as pool:
    ...     results = pool.map(some_function, data)
    >>>
    >>> # Non-deterministic (uses OS entropy)
    >>> with Pool(4, initializer=seed_worker) as pool:
    ...     results = pool.map(some_function, data)
    """
    if global_seed is None:
        # Non-deterministic: use OS entropy
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    else:
        # Deterministic: use global seed + worker ID
        worker_id = (
            mp.current_process()._identity[0] if mp.current_process()._identity else 0
        )
        seed = (global_seed + worker_id) % (2**32)
        np.random.seed(seed)


def create_seed_worker_initializer(global_seed: Optional[int] = None):
    """
    Create a worker initializer function with a specific global seed.

    This is a convenience function for creating partial functions
    suitable for use with multiprocessing.Pool.

    Parameters
    ----------
    global_seed : int, optional
        Global random seed

    Returns
    -------
    callable
        Function suitable for Pool initializer parameter

    Examples
    --------
    >>> from multiprocessing import Pool
    >>>
    >>> initializer = create_seed_worker_initializer(42)
    >>> with Pool(4, initializer=initializer) as pool:
    ...     results = pool.map(some_function, data)
    """
    from functools import partial

    return partial(seed_worker, global_seed)
