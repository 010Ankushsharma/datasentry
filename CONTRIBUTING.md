# Contributing to DataSentry

Thank you for your interest in contributing to DataSentry! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize the community's best interests

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/datasentry.git
   cd datasentry
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/010Ankushsharma/datasentry.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements-dev.txt
```

### Verify Setup

```bash
# Run tests
pytest

# Check code style
black --check src/datasentry tests
flake8 src/datasentry tests

# Run type checking
mypy src/datasentry
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please:

1. Check if the issue already exists
2. Update to the latest version to see if it's fixed

When reporting bugs, include:

- **Clear description** of the bug
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment details** (Python version, OS, package versions)
- **Code example** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. Check if the enhancement is already suggested
2. Provide clear use case and motivation
3. Describe expected behavior

### Pull Requests

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters maximum
- **Quotes**: Use double quotes for strings
- **Imports**: Group imports (stdlib, third-party, local)

### Code Formatting

We use automated tools to ensure consistency:

```bash
# Format code
black src/datasentry tests

# Sort imports
isort src/datasentry tests

# Check style
flake8 src/datasentry tests
```

### Type Hints

All functions should include type hints:

```python
def my_function(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None
) -> DetectionResult:
    """Function description.
    
    Args:
        X: Feature matrix.
        y: Target vector (optional).
    
    Returns:
        DetectionResult with analysis.
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def detect(
    self,
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None
) -> DetectionResult:
    """Detect data quality issues.
    
    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,), optional.
    
    Returns:
        DetectionResult containing detection results.
    
    Example:
        >>> detector = MyDetector()
        >>> result = detector.detect(X, y)
        >>> print(result.issue_detected)
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=datasentry --cov-report=html

# Run specific test file
pytest tests/test_detectors/test_imbalance.py

# Run with verbose output
pytest -v
```

### Writing Tests

All new functionality should include tests:

```python
# tests/test_detectors/test_my_detector.py
import pytest
from datasentry import MyDetector

class TestMyDetector:
    """Test suite for MyDetector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = MyDetector()
        assert detector.name == "MyDetector"
    
    def test_detect(self, sample_data):
        """Test detection functionality."""
        X, y = sample_data
        detector = MyDetector()
        result = detector.detect(X, y)
        assert result.detector_name == "MyDetector"
```

### Test Coverage

Aim for at least 90% test coverage for new code.

## Documentation

### Code Documentation

- All public functions must have docstrings
- Include type hints
- Provide usage examples

### README Updates

Update README.md if you:
- Add new features
- Change the API
- Add new dependencies

### Changelog

Add entries to CHANGELOG.md under the "Unreleased" section:

```markdown
## [Unreleased]

### Added
- New feature description

### Fixed
- Bug fix description
```

## Pull Request Process

1. **Update documentation** for any changed functionality

2. **Add tests** for new code

3. **Ensure all tests pass**:
   ```bash
   pytest
   ```

4. **Update CHANGELOG.md** with your changes

5. **Fill out the PR template** with:
   - Description of changes
   - Motivation
   - Testing performed
   - Breaking changes (if any)

6. **Wait for review** - maintainers will review your PR

7. **Address feedback** - make requested changes

8. **Merge** - once approved, your PR will be merged

## Commit Message Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

Examples:

```
feat: add new outlier detection method
fix: correct imbalance ratio calculation
docs: update API reference for detectors
test: add tests for missing value fixer
```

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub
- Contact the maintainers

Thank you for contributing to DataSentry!
