"""
Logging infrastructure for BAR_IMPACT.

This module provides a centralized logging configuration with sensible defaults.
Log levels can be controlled via environment variables.

Usage
-----
>>> from bar_impact.utils.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Processing started")
>>> logger.warning("Unusual value detected")
>>> logger.error("Processing failed", exc_info=True)

Environment Variables
--------------------
BAR_IMPACT_LOG_LEVEL : str
    Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    Default: INFO

BAR_IMPACT_LOG_FILE : str
    Path to log file. If not set, logs only go to console.

BAR_IMPACT_LOG_FORMAT : str
    Custom log format string. Default uses a structured format.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Default configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Package logger name
PACKAGE_NAME = "bar_impact"

# Track if root logger has been configured
_configured = False


def get_log_level() -> int:
    """Get log level from environment variable or default."""
    level_str = os.environ.get("BAR_IMPACT_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str, logging.INFO)


def get_log_format() -> str:
    """Get log format from environment variable or default."""
    return os.environ.get("BAR_IMPACT_LOG_FORMAT", DEFAULT_LOG_FORMAT)


def get_log_file() -> Optional[Path]:
    """Get log file path from environment variable if set."""
    log_file = os.environ.get("BAR_IMPACT_LOG_FILE")
    if log_file:
        return Path(log_file)
    return None


def configure_logging(
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configure the package-level logging.

    This function sets up the root logger for the bar_impact package.
    It's called automatically when get_logger() is first invoked.

    Parameters
    ----------
    level : int, optional
        Logging level. If not provided, uses BAR_IMPACT_LOG_LEVEL env var or INFO.
    log_file : Path, optional
        Path to log file. If not provided, uses BAR_IMPACT_LOG_FILE env var.
    log_format : str, optional
        Log format string. If not provided, uses BAR_IMPACT_LOG_FORMAT env var.
    force : bool, default False
        If True, reconfigure even if already configured.
    """
    global _configured

    if _configured and not force:
        return

    # Get configuration values
    level = level if level is not None else get_log_level()
    log_format = log_format if log_format is not None else get_log_format()
    log_file = log_file if log_file is not None else get_log_file()

    # Get package root logger
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=DEFAULT_DATE_FORMAT)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.

    This function ensures the logging infrastructure is configured
    before returning a logger. If name is None, returns the package
    root logger.

    Parameters
    ----------
    name : str, optional
        Logger name, typically __name__. If None, returns package root logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting processing")
    >>> logger.debug("Debug details: %s", details)
    """
    # Ensure logging is configured
    configure_logging()

    # Return appropriate logger
    if name is None:
        return logging.getLogger(PACKAGE_NAME)

    # If name starts with package name, use it directly
    if name.startswith(PACKAGE_NAME):
        return logging.getLogger(name)

    # Otherwise, make it a child of package logger
    return logging.getLogger(f"{PACKAGE_NAME}.{name}")


def set_log_level(level: int) -> None:
    """
    Set the log level for all bar_impact loggers.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).

    Examples
    --------
    >>> import logging
    >>> from bar_impact.utils.logging import set_log_level
    >>> set_log_level(logging.DEBUG)
    """
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging() -> None:
    """Disable all bar_impact logging (useful for tests)."""
    logging.getLogger(PACKAGE_NAME).setLevel(logging.CRITICAL + 1)


def enable_logging(level: Optional[int] = None) -> None:
    """Re-enable logging after disable_logging()."""
    level = level if level is not None else get_log_level()
    set_log_level(level)


class LoggingContext:
    """
    Context manager for temporarily changing log level.

    Examples
    --------
    >>> from bar_impact.utils.logging import get_logger, LoggingContext
    >>> import logging
    >>> logger = get_logger(__name__)
    >>> with LoggingContext(logging.DEBUG):
    ...     logger.debug("This will be logged")
    >>> logger.debug("This may not be logged")
    """

    def __init__(self, level: int):
        """
        Initialize context manager.

        Parameters
        ----------
        level : int
            Temporary log level to use within context.
        """
        self.level = level
        self.original_level: Optional[int] = None

    def __enter__(self):
        """Enter context and set temporary log level."""
        logger = logging.getLogger(PACKAGE_NAME)
        self.original_level = logger.level
        set_log_level(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original log level."""
        if self.original_level is not None:
            set_log_level(self.original_level)
        return False
