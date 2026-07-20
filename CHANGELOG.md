# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit hooks configuration for code quality enforcement
- GitHub Actions CI/CD pipeline for automated testing
- Logging infrastructure module (`bar_impact.utils.logging`)
- Custom exception hierarchy (`bar_impact.exceptions`)
- PEP 561 py.typed marker for type checking support
- CHANGELOG.md for tracking version history
- CONTRIBUTING.md with development guidelines
- LICENSE file (MIT)

### Changed
- Version management now uses `importlib.metadata` (single source of truth in pyproject.toml)
- Updated pyproject.toml with additional dev dependencies

### Fixed
- Added skipif markers for tests requiring optional dependencies (pymaster, jax, jaxili)

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Core data structures: `ConvergenceMap`, `SurveyMask`, `DataVector`
- Processing modules: `PowerSpectrumProcessor`, `L1NormProcessor`, `PeakCountProcessor`
- BNT transform support for tomographic analysis
- NPE inference module with JAX backend
- TARP coverage testing
- Comprehensive test suite with 205 test cases
- Documentation in docs/workflows/

[Unreleased]: https://github.com/AndreasTersenov/bar_impact/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AndreasTersenov/bar_impact/releases/tag/v0.1.0
