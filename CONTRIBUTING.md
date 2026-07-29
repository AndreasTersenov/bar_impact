# Contributing to BAR_IMPACT

Thank you for your interest in contributing to BAR_IMPACT! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AndreasTersenov/bar_impact.git
   cd bar_impact
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[all]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### External Dependencies

Some features require external packages not available on PyPI:

- **pycs** (CosmoStat): Required for wavelet L1 norm calculations
- **jaxili**: Required for NPE inference

## Development Workflow

### Code Style

This project uses:
- **black** for code formatting (line length: 88)
- **isort** for import sorting (black profile)
- **ruff** for linting
- **mypy** for type checking

Pre-commit hooks will automatically check these on each commit.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_core.py -v

# Run tests with coverage
pytest tests/ --cov=bar_impact --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_power_spectrum"
```

### Type Checking

```bash
mypy src/bar_impact/
```

### Manual Code Quality Checks

```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Run linter
ruff check src/ scripts/ tests/
```

## Making Changes

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions/changes

### Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when applicable

Example:
```
Add shape noise validation in ConvergenceMap

- Validate sigma_e is positive
- Add warning for unusually high noise values
- Update tests to cover edge cases

Fixes #123
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes following the code style guidelines
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Update documentation if needed
6. Submit a pull request with a clear description

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How the changes were tested

## Related Issues
Fixes #issue_number
```

## Code Guidelines

### Module Structure

Follow the existing package structure:
- `core/` - Data structures and fundamental classes
- `processing/` - Summary statistic processors
- `inference/` - NPE and coverage testing
- `analysis/` - Result aggregation and visualization
- `utils/` - Utility functions and helpers

### Adding New Processors

New summary statistic processors should follow the existing pattern:
1. Create a `*Config` dataclass for configuration
2. Subclass `BaseProcessor`
3. Implement `process_single()` method
4. Add a `compute_*()` convenience function
5. Export from `processing/__init__.py`

### Error Handling

Use the custom exception hierarchy in `bar_impact.exceptions`:
- `ConfigurationError` - Invalid configuration
- `ProcessingError` - Processing failures
- `MaskError` - Mask-related issues
- `InferenceError` - Inference failures
- `DataLoadError` - Data loading issues

### Logging

Use the logging module instead of print statements:
```python
from bar_impact.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.warning("Unusual value detected")
logger.error("Processing failed", exc_info=True)
```

## Testing Guidelines

### Test Organization

- `tests/unit/` - Unit tests for individual functions/classes
- `tests/integration/` - Integration tests for workflows
- `tests/test_*.py` - Module-level tests

### Writing Tests

- Use pytest fixtures for common setup
- Mark tests requiring optional dependencies with `@pytest.mark.skipif`
- Use meaningful test names: `test_power_spectrum_handles_nan_values`
- Include docstrings explaining what is being tested

Example:
```python
@pytest.mark.skipif(not HAS_NAMASTER, reason="pymaster not installed")
def test_namaster_mode_coupling():
    """Test NaMaster mode coupling correction is applied correctly."""
    ...
```

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.
