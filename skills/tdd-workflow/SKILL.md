---
name: tdd-workflow
description: Use this skill when writing new features, fixing bugs, or refactoring code. Enforces test-driven development with 80%+ coverage including unit, integration, and numerical validation tests.
---

# Test-Driven Development Workflow

This skill ensures all code development follows TDD principles with comprehensive test coverage for scientific Python code.

## When to Activate

- Writing new features or functionality
- Fixing bugs or issues
- Refactoring existing code
- Adding new algorithms
- Creating new array operations

## Core Principles

### 1. Tests BEFORE Code
ALWAYS write tests first, then implement code to make tests pass.

### 2. Coverage Requirements
- Minimum 80% coverage (unit + integration + E2E)
- All edge cases covered
- Error scenarios tested
- Boundary conditions verified

### 3. Test Types

#### Unit Tests
- Individual functions and utilities
- Array operations
- Pure functions
- Mathematical computations

#### Integration Tests
- Multi-step processing pipelines
- File I/O operations
- Backend compatibility (NumPy, JAX, CuPy)
- Algorithm combinations

#### Numerical Validation Tests
- Precision against reference implementations
- Edge cases (NaN, Inf, empty arrays)
- Performance benchmarks
- Property-based tests

## TDD Workflow Steps

### Step 1: Define Requirements
```
As a [role], I want to [action], so that [benefit]

Example:
As a researcher, I want to filter ultrasound signals with a bandpass filter,
so that I can isolate frequencies of interest for analysis.
```

### Step 2: Generate Test Cases
For each requirement, create comprehensive test cases:

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestBandpassFilter:
    def test_preserves_passband_frequencies(self):
        """Filter preserves frequencies in passband."""
        # Test implementation
        pass
    
    def test_handles_empty_signal(self):
        """Reject empty signal gracefully."""
        # Test edge case
        pass
    
    def test_rejects_invalid_cutoff_frequencies(self):
        """Raise error for invalid cutoff frequencies."""
        # Test validation
        pass
    
    def test_preserves_signal_shape(self):
        """Output shape matches input shape."""
        # Test shape preservation
        pass
```

### Step 3: Run Tests (They Should Fail)
```bash
pytest tests/test_filtering.py
# Tests should fail - we haven't implemented yet
```

### Step 4: Implement Code
Write minimal code to make tests pass:

```python
from jaxtyping import Float
from numpy.typing import NDArray

def bandpass_filter(
    signal: Float[NDArray, "time"],
    lowcut: float,
    highcut: float,
    fs: float
) -> Float[NDArray, "time"]:
    """Apply bandpass filter to signal."""
    # Implementation guided by tests
    pass
```

### Step 5: Run Tests Again
```bash
pytest tests/test_filtering.py
# Tests should now pass
```

### Step 6: Refactor
Improve code quality while keeping tests green:
- Remove duplication
- Improve naming
- Optimize performance
- Enhance readability

### Step 7: Verify Coverage
```bash
pytest --cov=src --cov-report=term-missing
# Verify 80%+ coverage achieved
```

## Testing Patterns

### Unit Test Pattern (pytest)
```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestSignalFiltering:
    """Test signal filtering functions."""
    
    def test_filters_noise_from_signal(self):
        """Filter removes noise while preserving signal."""
        # Arrange
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 50 * t)
        noisy = signal + np.random.normal(0, 0.1, len(signal))
        
        # Act
        filtered = bandpass_filter(noisy, lowcut=40, highcut=60, fs=fs)
        
        # Assert
        noise_reduction = np.std(filtered - signal) / np.std(noisy - signal)
        assert noise_reduction < 0.5
    
    def test_preserves_signal_shape(self):
        """Output shape matches input shape."""
        signal = np.random.randn(1000)
        filtered = bandpass_filter(signal, lowcut=40, highcut=60, fs=1000)
        assert filtered.shape == signal.shape
    
    def test_rejects_empty_signal(self):
        """Raise error for empty signal."""
        with pytest.raises(ValueError, match="cannot be empty"):
            bandpass_filter(np.array([]), lowcut=40, highcut=60, fs=1000)
```

### Integration Test Pattern
```python
import pytest
import numpy as np
from pathlib import Path

class TestProcessingPipeline:
    """Test multi-step signal processing pipeline."""
    
    def test_full_pipeline_with_file_io(self, tmp_path):
        """Test complete processing from file to file."""
        # Arrange: Create test input file
        input_file = tmp_path / "input.npy"
        test_signal = np.random.randn(1000)
        np.save(input_file, test_signal)
        
        # Act: Run processing pipeline
        output_file = tmp_path / "output.npy"
        process_signal_file(input_file, output_file, lowcut=40, highcut=60)
        
        # Assert: Verify output file exists and has correct data
        assert output_file.exists()
        result = np.load(output_file)
        assert result.shape == test_signal.shape
    
    def test_backend_compatibility(self):
        """Test algorithm works with NumPy and JAX."""
        signal = np.random.randn(100)
        
        # NumPy backend
        result_numpy = compute_fft(signal)
        
        # JAX backend
        import jax.numpy as jnp
        signal_jax = jnp.array(signal)
        result_jax = compute_fft(signal_jax)
        
        # Results should be equivalent
        assert_allclose(result_numpy, np.array(result_jax), rtol=1e-6)
    
    def test_handles_file_not_found(self):
        """Gracefully handle missing input file."""
        with pytest.raises(FileNotFoundError):
            process_signal_file(Path("nonexistent.npy"), Path("output.npy"))
```

### Numerical Validation Test Pattern
```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestNumericalPrecision:
    """Test numerical accuracy against reference implementation."""
    
    @pytest.fixture
    def reference_data(self):
        """Load PyMUST reference data."""
        return np.load("tests/data/reference_pfield.npz")
    
    def test_matches_pymust_reference(self, reference_data):
        """Results match PyMUST within tolerance."""
        result = compute_pfield(
            reference_data["positions"],
            reference_data["frequency"]
        )
        
        assert_allclose(
            result,
            reference_data["expected"],
            rtol=1e-4,  # 0.01% relative tolerance
            atol=1e-8   # Absolute tolerance for near-zero
        )
    
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_scaling(self, benchmark, size):
        """Benchmark performance at different sizes."""
        signal = np.random.randn(size)
        result = benchmark(compute_fft, signal)
        assert result.shape == signal.shape
```

## Test File Organization

```
src/
├── my_library/
│   ├── __init__.py
│   ├── filtering.py
│   ├── transforms.py
│   └── beamforming.py
tests/
├── test_filtering.py          # Unit tests for filtering
├── test_transforms.py         # Unit tests for transforms
├── test_beamforming.py        # Unit tests for beamforming
├── test_integration.py        # Integration tests
├── conftest.py                # pytest fixtures
└── data/
    ├── reference_pfield.npz   # Reference data
    └── test_signals.npz       # Test data
```

## Mocking and Fixtures

### File I/O Mocking
```python
from unittest.mock import Mock, patch, mock_open

def test_load_signal_with_mock(tmp_path):
    """Test loading signal with temporary file."""
    # Create temporary test file
    test_file = tmp_path / "signal.npy"
    test_data = np.array([1.0, 2.0, 3.0])
    np.save(test_file, test_data)
    
    # Test loading
    result = load_signal(test_file)
    assert_array_equal(result, test_data)

@patch('builtins.open', mock_open(read_data='1.0,2.0,3.0'))
def test_load_csv_with_mock():
    """Test loading CSV with mocked file."""
    result = load_csv_signal('dummy.csv')
    assert len(result) == 3
```

### Fixtures for Test Data
```python
import pytest

@pytest.fixture
def sample_signal():
    """Generate sample 1D signal."""
    fs = 1000.0
    t = np.linspace(0, 1, int(fs))
    return np.sin(2 * np.pi * 50 * t)

@pytest.fixture
def reference_data():
    """Load reference data from PyMUST."""
    return np.load("tests/data/reference_pfield.npz")

class TestWithFixtures:
    def test_uses_sample_signal(self, sample_signal):
        """Test using fixture."""
        result = process_signal(sample_signal)
        assert result.shape == sample_signal.shape
```

### Backend Mocking
```python
@pytest.fixture(params=["numpy", "jax"])
def array_backend(request):
    """Test with multiple backends."""
    backend = request.param
    if backend == "numpy":
        import numpy as xp
    elif backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as xp
    return xp

def test_backend_compatibility(array_backend):
    """Algorithm works with all backends."""
    xp = array_backend
    arr = xp.array([1.0, 2.0, 3.0])
    result = compute_norm(arr)
    assert float(result) == pytest.approx(3.7416573867739413)
```

## Test Coverage Verification

### Run Coverage Report
```bash
pytest --cov=src --cov-report=term-missing
```

### Coverage Configuration
```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]

[tool.coverage.html]
directory = "htmlcov"
```

## Common Testing Mistakes to Avoid

### ❌ WRONG: Testing Implementation Details
```python
# Don't test internal implementation
assert processor._internal_buffer == expected
```

### ✅ CORRECT: Test Behavior
```python
# Test observable behavior
result = processor.process(signal)
assert result.shape == expected_shape
assert_allclose(result, expected_output, rtol=1e-4)
```

### ❌ WRONG: Using Exact Equality for Floats
```python
# Breaks due to floating point precision
assert result == 3.14159265359
```

### ✅ CORRECT: Use Tolerance
```python
# Use appropriate tolerance
assert result == pytest.approx(3.14159265359, rel=1e-10)
# Or for arrays
assert_allclose(result, expected, rtol=1e-4, atol=1e-8)
```

### ❌ WRONG: No Test Isolation
```python
# Tests depend on each other
def test_creates_signal():
    global test_signal
    test_signal = generate_signal()

def test_processes_signal():
    result = process(test_signal)  # Depends on previous test
```

### ✅ CORRECT: Independent Tests
```python
# Each test sets up its own data
def test_creates_signal():
    signal = generate_signal()
    assert signal.shape == (1000,)

def test_processes_signal():
    signal = generate_signal()  # Independent setup
    result = process(signal)
    assert result.shape == signal.shape
```

## Continuous Testing

### Watch Mode During Development
```bash
pytest-watch
# Or use pytest-xdist for parallel execution
pytest -n auto --looponfail
```

### Pre-Commit Hook
```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

### CI/CD Integration
```yaml
# GitHub Actions
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run tests with coverage
        run: |
          uv run pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Best Practices

1. **Write Tests First** - Always TDD
2. **One Assert Per Test** - Focus on single behavior
3. **Descriptive Test Names** - Explain what's tested
4. **Arrange-Act-Assert** - Clear test structure
5. **Mock External Dependencies** - Isolate unit tests
6. **Test Edge Cases** - Null, undefined, empty, large
7. **Test Error Paths** - Not just happy paths
8. **Keep Tests Fast** - Unit tests < 50ms each
9. **Clean Up After Tests** - No side effects
10. **Review Coverage Reports** - Identify gaps

## Success Metrics

- 80%+ code coverage achieved
- All tests passing (green)
- No skipped or disabled tests
- Fast test execution (< 30s for unit tests)
- E2E tests cover critical user flows
- Tests catch bugs before production

---

**Remember**: Tests are not optional. They are the safety net that enables confident refactoring, rapid development, and production reliability.
