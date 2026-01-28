"""
Custom exceptions for BAR_IMPACT.

This module provides a hierarchy of custom exceptions for better error handling
and debugging. Using specific exception types allows callers to catch and handle
different failure modes appropriately.

Exception Hierarchy
-------------------
BarImpactError
    Base exception for all bar_impact errors.

    ConfigurationError
        Raised for invalid configuration parameters.

    DataLoadError
        Raised when data loading fails (missing files, corrupt data).

    ProcessingError
        Raised during processing failures.

        MaskError
            Raised for mask-related issues (incompatible sizes, invalid values).

        TransformError
            Raised for BNT or other transform failures.

    InferenceError
        Raised during NPE or other inference failures.

        TrainingError
            Raised when model training fails.

        SamplingError
            Raised when posterior sampling fails.

Usage
-----
>>> from bar_impact.exceptions import ConfigurationError, ProcessingError
>>>
>>> # Raise specific exception
>>> raise ConfigurationError("lmax must be positive", parameter="lmax", value=-1)
>>>
>>> # Catch bar_impact exceptions
>>> try:
...     process_data()
... except ProcessingError as e:
...     logger.error(f"Processing failed: {e}")
"""

from typing import Any, Optional


class BarImpactError(Exception):
    """
    Base exception for all bar_impact errors.

    All custom exceptions in this package inherit from this class,
    allowing callers to catch all bar_impact-related errors with a single
    except clause when needed.

    Parameters
    ----------
    message : str
        Human-readable error message.
    **context : Any
        Additional context key-value pairs for debugging.

    Attributes
    ----------
    message : str
        The error message.
    context : dict
        Additional context information.
    """

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ConfigurationError(BarImpactError):
    """
    Raised for invalid configuration parameters.

    Use this when configuration values are out of range, incompatible,
    or otherwise invalid.

    Parameters
    ----------
    message : str
        Description of the configuration error.
    parameter : str, optional
        Name of the invalid parameter.
    value : Any, optional
        The invalid value that was provided.
    expected : str, optional
        Description of what was expected.

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "lmax must be positive",
    ...     parameter="lmax",
    ...     value=-1,
    ...     expected="positive integer"
    ... )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(
            message, parameter=parameter, value=value, expected=expected, **context
        )
        self.parameter = parameter
        self.value = value
        self.expected = expected


class DataLoadError(BarImpactError):
    """
    Raised when data loading fails.

    Use this for missing files, corrupt data, unexpected formats,
    or other I/O related errors during data loading.

    Parameters
    ----------
    message : str
        Description of the loading error.
    file_path : str, optional
        Path to the file that failed to load.
    reason : str, optional
        Additional details about why loading failed.

    Examples
    --------
    >>> raise DataLoadError(
    ...     "Failed to load convergence map",
    ...     file_path="/path/to/map.h5",
    ...     reason="missing 'kappa' dataset"
    ... )
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        reason: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, file_path=file_path, reason=reason, **context)
        self.file_path = file_path
        self.reason = reason


class ProcessingError(BarImpactError):
    """
    Raised during processing failures.

    Base exception for errors that occur during data processing,
    including summary statistic computation.

    Parameters
    ----------
    message : str
        Description of the processing error.
    step : str, optional
        Processing step where the error occurred.
    input_shape : tuple, optional
        Shape of the input data that caused the error.

    Examples
    --------
    >>> raise ProcessingError(
    ...     "NaN values detected in power spectrum",
    ...     step="power_spectrum",
    ...     input_shape=(12582912,)
    ... )
    """

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **context: Any,
    ):
        super().__init__(message, step=step, input_shape=input_shape, **context)
        self.step = step
        self.input_shape = input_shape


class MaskError(ProcessingError):
    """
    Raised for mask-related issues.

    Use this for incompatible mask sizes, invalid mask values,
    or other mask-specific errors.

    Parameters
    ----------
    message : str
        Description of the mask error.
    mask_shape : tuple, optional
        Shape of the mask that caused the error.
    map_shape : tuple, optional
        Shape of the map the mask was applied to.
    invalid_values : int, optional
        Number of invalid values in the mask.

    Examples
    --------
    >>> raise MaskError(
    ...     "Mask and map have incompatible sizes",
    ...     mask_shape=(3145728,),
    ...     map_shape=(12582912,)
    ... )
    """

    def __init__(
        self,
        message: str,
        mask_shape: Optional[tuple] = None,
        map_shape: Optional[tuple] = None,
        invalid_values: Optional[int] = None,
        **context: Any,
    ):
        super().__init__(
            message,
            step="masking",
            mask_shape=mask_shape,
            map_shape=map_shape,
            invalid_values=invalid_values,
            **context,
        )
        self.mask_shape = mask_shape
        self.map_shape = map_shape
        self.invalid_values = invalid_values


class TransformError(ProcessingError):
    """
    Raised for BNT or other transform failures.

    Use this when mathematical transforms (BNT, wavelet, etc.) fail
    due to invalid inputs or numerical issues.

    Parameters
    ----------
    message : str
        Description of the transform error.
    transform : str, optional
        Name of the transform that failed.
    matrix_shape : tuple, optional
        Shape of the transform matrix if applicable.

    Examples
    --------
    >>> raise TransformError(
    ...     "BNT matrix shape incompatible with number of bins",
    ...     transform="BNT",
    ...     matrix_shape=(4, 4),
    ...     n_bins=5
    ... )
    """

    def __init__(
        self,
        message: str,
        transform: Optional[str] = None,
        matrix_shape: Optional[tuple] = None,
        **context: Any,
    ):
        super().__init__(
            message, step=transform or "transform", matrix_shape=matrix_shape, **context
        )
        self.transform = transform
        self.matrix_shape = matrix_shape


class InferenceError(BarImpactError):
    """
    Raised during NPE or other inference failures.

    Base exception for errors that occur during inference,
    including training and sampling.

    Parameters
    ----------
    message : str
        Description of the inference error.
    method : str, optional
        Inference method (e.g., "NPE", "ABC").
    iteration : int, optional
        Training iteration where the error occurred.

    Examples
    --------
    >>> raise InferenceError(
    ...     "Inference failed to converge",
    ...     method="NPE",
    ...     iteration=1000
    ... )
    """

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        iteration: Optional[int] = None,
        **context: Any,
    ):
        super().__init__(message, method=method, iteration=iteration, **context)
        self.method = method
        self.iteration = iteration


class TrainingError(InferenceError):
    """
    Raised when model training fails.

    Use this for training-specific failures such as NaN losses,
    divergence, or resource exhaustion.

    Parameters
    ----------
    message : str
        Description of the training error.
    epoch : int, optional
        Training epoch where the error occurred.
    loss : float, optional
        Loss value at failure (may be NaN/inf).

    Examples
    --------
    >>> raise TrainingError(
    ...     "Training loss became NaN",
    ...     epoch=50,
    ...     loss=float('nan')
    ... )
    """

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        **context: Any,
    ):
        super().__init__(message, epoch=epoch, loss=loss, **context)
        self.epoch = epoch
        self.loss = loss


class SamplingError(InferenceError):
    """
    Raised when posterior sampling fails.

    Use this for sampling-specific failures such as divergent chains,
    numerical instability, or invalid posterior shapes.

    Parameters
    ----------
    message : str
        Description of the sampling error.
    n_samples : int, optional
        Number of samples requested.
    n_obtained : int, optional
        Number of samples successfully obtained.

    Examples
    --------
    >>> raise SamplingError(
    ...     "Posterior sampling produced invalid samples",
    ...     n_samples=10000,
    ...     n_obtained=8500,
    ...     reason="divergent chains"
    ... )
    """

    def __init__(
        self,
        message: str,
        n_samples: Optional[int] = None,
        n_obtained: Optional[int] = None,
        **context: Any,
    ):
        super().__init__(
            message, n_samples=n_samples, n_obtained=n_obtained, **context
        )
        self.n_samples = n_samples
        self.n_obtained = n_obtained
