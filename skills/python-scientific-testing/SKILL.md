---
name: python-scientific-testing
description: Python scientific testing patterns including parametrized tests, fixtures, benchmarks, property-based testing with hypothesis, and numerical validation. Follows TDD methodology with pytest.
---

# Python Scientific Testing Patterns

Comprehensive Python testing patterns for writing reliable, maintainable tests for scientific computing code following TDD methodology.

## When to Activate

- Writing new Python scientific functions
- Adding test coverage to numerical code
- Creating benchmarks for performance-critical algorithms
- Implementing property-based tests for array operations
- Following TDD workflow in Python scientific projects

## TDD Workflow for Python Scientific Code

### The RED-GREEN-REFACTOR Cycle

```
RED     → Write a failing test first
GREEN   → Write minimal code to pass the test
REFACTOR → Improve code while keeping tests green
REPEAT  → Continue with next requirement
```

### Step-by-Step TDD in Python

```python
# Step 1: Define the interface/signature
# signal_processing.py
from jaxtyping import Float
from numpy.typing import NDArray

def bandpass_filter(
    signal: Float[NDArray, "time"],
    lowcut: float,
    highcut: float,
    fs: float
) -> Float[NDArray, "time"]:
    """Apply bandpass filter to signal."""
    raise NotImplementedError("not implemented")

# Step 2: Write failing test (RED)
# test_signal_processing.py
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestBandpassFilter:
    def test_filters_frequencies_in_passband(self):
        """Filter preserves frequencies in passband."""
        # Arrange
        fs = 1000.0  # Hz
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz signal
        
        # Act
        filtered = bandpass_filter(signal, lowcut=40, highcut=60, fs=fs)
        
        # Assert
        assert_allclose(filtered, signal, rtol=0.1)

# Step 3: Run test - verify FAIL
# $ pytest test_signal_processing.py
# FAILED - NotImplementedError: not implemented

# Step 4: Implement minimal code (GREEN)
from scipy import signal as sp_signal

def bandpass_filter(
    signal: Float[NDArray, "time"],
    lowcut: float,
    highcut: float,
    fs: float
) -> Float[NDArray, "time"]:
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sp_signal.butter(4, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, signal)

# Step 5: Run test - verify PASS
# $ pytest test_signal_processing.py
# PASSED

# Step 6: Refactor if needed, verify tests still pass
```

## Parametrized Tests

The standard pattern for pytest. Enables comprehensive coverage with minimal code.

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestComputeDelay:
    @pytest.mark.parametrize("distance,speed,expected", [
        (0.03, 1540.0, 0.03/1540.0),  # Typical tissue
        (0.01, 1480.0, 0.01/1480.0),  # Water
        (0.05, 1540.0, 0.05/1540.0),  # Deeper tissue
        (0.0, 1540.0, 0.0),           # Zero distance
    ])
    def test_compute_delay_values(self, distance, speed, expected):
        """Compute delay for various distances and speeds."""
        result = compute_delay(distance, speed)
        assert_allclose(result, expected, rtol=1e-10)
    
    @pytest.mark.parametrize("distance,speed", [
        (-0.01, 1540.0),  # Negative distance
        (0.01, -1540.0),  # Negative speed
        (0.01, 0.0),      # Zero speed
    ])
    def test_compute_delay_invalid_inputs(self, distance, speed):
        """Reject invalid inputs."""
        with pytest.raises(ValueError):
            compute_delay(distance, speed)
```

### Parametrized Tests with Array Shapes

```python
class TestArrayOperations:
    @pytest.mark.parametrize("shape", [
        (100,),           # 1D
        (10, 10),         # 2D square
        (5, 20),          # 2D rectangular
        (2, 3, 4),        # 3D
    ])
    def test_normalize_preserves_shape(self, shape):
        """Normalization preserves array shape."""
        arr = np.random.randn(*shape)
        result = normalize(arr)
        assert result.shape == shape
    
    @pytest.mark.parametrize("dtype", [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ])
    def test_fft_preserves_dtype_precision(self, dtype):
        """FFT preserves dtype precision class."""
        signal = np.random.randn(100).astype(dtype)
        result = compute_fft(signal)
        assert result.dtype == np.result_type(dtype, np.complex64)
```

## Fixtures

### Basic Fixtures

```python
import pytest
import numpy as np

@pytest.fixture
def sample_signal():
    """Generate sample 1D signal."""
    fs = 1000.0
    t = np.linspace(0, 1, int(fs))
    return np.sin(2 * np.pi * 50 * t)

@pytest.fixture
def transducer_params():
    """Standard transducer parameters."""
    return TransducerParams(
        fc=5e6,
        pitch=0.3e-3,
        n_elements=128,
        width=0.27e-3
    )

class TestSignalProcessing:
    def test_filter_removes_noise(self, sample_signal):
        """Filter removes high-frequency noise."""
        noisy = sample_signal + np.random.normal(0, 0.1, len(sample_signal))
        filtered = lowpass_filter(noisy, cutoff=100, fs=1000)
        
        # Check noise reduction
        noise_before = np.std(noisy - sample_signal)
        noise_after = np.std(filtered - sample_signal)
        assert noise_after < noise_before * 0.5
```

### Parametrized Fixtures

```python
@pytest.fixture(params=["numpy", "jax", "cupy"])
def array_backend(request):
    """Test with multiple array backends."""
    backend = request.param
    if backend == "numpy":
        import numpy as xp
    elif backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as xp
    elif backend == "cupy":
        pytest.importorskip("cupy")
        import cupy as xp
    return xp

def test_backend_compatibility(array_backend):
    """Algorithm works with all backends."""
    xp = array_backend
    arr = xp.array([1.0, 2.0, 3.0])
    result = compute_norm(arr)
    assert float(result) == pytest.approx(3.7416573867739413)
```

### Fixture Scope

```python
import zarr

@pytest.fixture(scope="session")
def reference_data():
    """Load reference data once per test session."""
    # Expensive operation - load reference data from Zarr
    return zarr.open("tests/data/reference_results.zarr", mode="r")

@pytest.fixture(scope="module")
def large_test_array():
    """Generate large array once per module."""
    return np.random.randn(10000, 1000)

@pytest.fixture(scope="function")  # Default
def temp_array():
    """Fresh array for each test."""
    return np.zeros(100)
```

## Numerical Precision Testing

### Using numpy.testing

```python
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal
)
import zarr

class TestNumericalPrecision:
    def test_matches_reference_implementation(self, reference_data):
        """Results match PyMUST reference within tolerance."""
        result = compute_pfield(
            reference_data["positions"][:],
            reference_data["frequency"][:]
        )
        assert_allclose(
            result,
            reference_data["expected"][:],
            rtol=1e-4,  # 0.01% relative tolerance
            atol=1e-8   # Absolute tolerance for near-zero values
        )
    
    def test_integer_array_equality(self):
        """Integer arrays must match exactly."""
        indices = compute_indices(10, 5)
        expected = np.array([0, 2, 4, 6, 8])
        assert_array_equal(indices, expected)
    
    def test_floating_point_almost_equal(self):
        """Floating point comparison with decimal places."""
        result = compute_ratio(1.0, 3.0)
        assert_array_almost_equal(result, 0.333333, decimal=6)
```

### Custom Tolerance Testing

```python
def assert_signals_close(
    actual: NDArray,
    expected: NDArray,
    rtol: float = 1e-4,
    atol: float = 1e-8,
    name: str = "signal"
) -> None:
    """Assert signals are close with informative error."""
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Add diagnostic information
        max_diff = np.max(np.abs(actual - expected))
        rel_diff = max_diff / (np.max(np.abs(expected)) + 1e-10)
        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Max absolute difference: {max_diff:.2e}\n"
            f"  Max relative difference: {rel_diff:.2%}\n"
            f"  Required rtol: {rtol:.2e}, atol: {atol:.2e}\n"
            f"Original error: {e}"
        )
```

## Edge Case Testing

### Array Edge Cases

```python
class TestEdgeCases:
    def test_empty_array(self):
        """Handle empty array gracefully."""
        with pytest.raises(ValueError, match="cannot be empty"):
            process_signal(np.array([]))
    
    def test_single_element(self):
        """Handle single-element array."""
        result = process_signal(np.array([1.0]))
        assert result.shape == (1,)
    
    def test_nan_values(self):
        """Reject arrays with NaN."""
        signal = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="contains NaN"):
            process_signal(signal)
    
    def test_inf_values(self):
        """Reject arrays with Inf."""
        signal = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="contains Inf"):
            process_signal(signal)
    
    def test_large_array(self):
        """Handle large arrays without memory error."""
        large_signal = np.random.randn(10_000_000)
        result = process_signal(large_signal)
        assert result.shape == large_signal.shape
    
    @pytest.mark.parametrize("ndim", [0, 1, 2, 3, 4])
    def test_various_dimensions(self, ndim):
        """Handle various array dimensions."""
        shape = tuple([10] * ndim) if ndim > 0 else ()
        arr = np.random.randn(*shape) if ndim > 0 else np.array(1.0)
        
        if ndim == 1:
            result = process_signal(arr)
            assert result.ndim == 1
        else:
            with pytest.raises(ValueError, match="Expected 1D"):
                process_signal(arr)
```

## Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

class TestPropertyBased:
    @given(arrays(
        dtype=np.float64,
        shape=st.integers(min_value=10, max_value=1000),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    ))
    def test_normalize_properties(self, signal):
        """Normalized signal has zero mean and unit variance."""
        normalized = normalize(signal)
        
        # Property 1: Mean is close to zero
        assert abs(normalized.mean()) < 1e-10
        
        # Property 2: Std is close to one
        assert abs(normalized.std() - 1.0) < 1e-10
        
        # Property 3: Shape is preserved
        assert normalized.shape == signal.shape
    
    @given(
        st.floats(min_value=0.001, max_value=1.0),  # distance
        st.floats(min_value=1000.0, max_value=2000.0)  # speed
    )
    def test_delay_monotonic(self, distance, speed):
        """Delay increases monotonically with distance."""
        delay1 = compute_delay(distance, speed)
        delay2 = compute_delay(distance * 2, speed)
        assert delay2 > delay1
    
    @given(arrays(
        dtype=np.float64,
        shape=(100,),
        elements=st.floats(min_value=-10, max_value=10)
    ))
    def test_fft_invertible(self, signal):
        """FFT is invertible."""
        spectrum = np.fft.fft(signal)
        reconstructed = np.fft.ifft(spectrum).real
        assert_allclose(reconstructed, signal, rtol=1e-10)
```

## Benchmarks with pytest-benchmark

```python
import pytest

class TestPerformance:
    def test_numpy_baseline(self, benchmark):
        """Benchmark NumPy implementation."""
        signal = np.random.randn(10000)
        result = benchmark(compute_fft_numpy, signal)
        assert result.shape == signal.shape
    
    def test_jax_speedup(self, benchmark):
        """Benchmark JAX implementation."""
        pytest.importorskip("jax")
        signal = np.random.randn(10000)
        result = benchmark(compute_fft_jax, signal)
        assert result.shape == signal.shape
    
    @pytest.mark.parametrize("size", [100, 1000, 10000, 100000])
    def test_scaling(self, benchmark, size):
        """Test performance scaling with size."""
        signal = np.random.randn(size)
        benchmark(compute_fft, signal)

# Run: pytest test_performance.py --benchmark-only
# Compare: pytest test_performance.py --benchmark-compare
```

### Benchmark Groups

```python
@pytest.mark.benchmark(group="fft")
def test_numpy_fft(benchmark):
    signal = np.random.randn(10000)
    benchmark(np.fft.fft, signal)

@pytest.mark.benchmark(group="fft")
def test_scipy_fft(benchmark):
    signal = np.random.randn(10000)
    benchmark(scipy.fft.fft, signal)

@pytest.mark.benchmark(group="filtering")
def test_bandpass_filter(benchmark):
    signal = np.random.randn(10000)
    benchmark(bandpass_filter, signal, 40, 60, 1000)
```

## Test Coverage

### Running Coverage

```bash
# Basic coverage
pytest --cov=src

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80

# Coverage for specific module
pytest --cov=src.signal_processing tests/test_signal.py
```

### Coverage Targets

| Code Type | Target |
|-----------|--------|
| Core algorithms | 95%+ |
| Public APIs | 90%+ |
| Utilities | 80%+ |
| Generated code | Exclude |

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if DEBUG:",
]
```

## Mocking and Fixtures for I/O

### Loading Reference Data with Zarr

```python
import zarr
import numpy as np
import pytest

class TestComputeDelays:
    """Test acoustic delay computation."""
    
    @pytest.fixture
    def reference_data(self):
        """Load PyMUST reference data from Zarr."""
        return zarr.open("tests/data/reference_delays.zarr", mode="r")
    
    def test_matches_pymust_reference(self, reference_data):
        """Verify delays match PyMUST within tolerance."""
        result = compute_delays(
            focus=reference_data["focus"][:],
            elements=reference_data["elements"][:]
        )
        
        assert_allclose(
            result,
            reference_data["expected"][:],
            rtol=1e-4,  # 0.01% relative tolerance
            atol=1e-8   # Absolute tolerance for near-zero
        )
    
    @pytest.mark.parametrize("speed", [-1540.0, 0.0])
    def test_rejects_invalid_speed(self, speed):
        """Reject non-positive speed values."""
        import array_api_compat
        xp = array_api_compat.get_namespace("numpy")
        focus = xp.array([0.0, 0.0, 0.03])
        elements = xp.array([[0.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="speed must be positive"):
            compute_delays(focus, elements, speed=speed)
    
    def test_handles_nan_in_focus(self):
        """Reject NaN in focus point."""
        import array_api_compat
        xp = array_api_compat.get_namespace("numpy")
        focus = xp.array([0.0, xp.nan, 0.03])
        elements = xp.array([[0.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="contain NaN"):
            compute_delays(focus, elements)
    
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_backend_compatibility(self, backend):
        """Algorithm works with NumPy and JAX."""
        import array_api_compat
        if backend == "numpy":
            xp = array_api_compat.get_namespace("numpy")
        else:
            pytest.importorskip("jax")
            xp = array_api_compat.get_namespace("jax")
        
        focus = xp.array([0.0, 0.0, 0.03])
        elements = xp.array([[0.0, 0.0, 0.0]])
        result = compute_delays(focus, elements)
        
        assert result.shape == (1,)
```

### Mocking File I/O

```python
from unittest.mock import Mock, patch, mock_open
import zarr

def test_load_signal_file(tmp_path):
    """Test loading signal from Zarr file."""
    # Create temporary test file
    test_file = tmp_path / "signal.zarr"
    test_data = np.array([1.0, 2.0, 3.0])
    zarr.save(test_file, test_data)
    
    # Test loading
    result = load_signal(test_file)
    assert_array_equal(result, test_data)

def test_save_signal_file(tmp_path):
    """Test saving signal to Zarr file."""
    test_file = tmp_path / "output.zarr"
    test_data = np.array([1.0, 2.0, 3.0])
    
    save_signal(test_file, test_data)
    
    # Verify file was created and contains correct data
    assert test_file.exists()
    loaded = zarr.load(test_file)
    assert_array_equal(loaded, test_data)

@patch('builtins.open', mock_open(read_data='1.0,2.0,3.0'))
def test_load_csv():
    """Test loading CSV with mocked file."""
    result = load_csv_signal('dummy.csv')
    assert len(result) == 3
```

## Testing Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_signal.py

# Run specific test
pytest tests/test_signal.py::TestBandpass::test_filters_noise

# Run tests matching pattern
pytest -k "bandpass"

# Run tests with markers
pytest -m "slow"
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run benchmarks
pytest --benchmark-only

# Run with multiple workers (parallel)
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Rerun failed tests
pytest --lf

# Run tests that failed last time, then all
pytest --ff
```

## Markers for Test Organization

```python
import pytest

@pytest.mark.slow
def test_large_computation():
    """Mark slow tests."""
    result = expensive_computation()
    assert result is not None

@pytest.mark.gpu
@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
def test_gpu_acceleration():
    """Mark GPU tests."""
    result = compute_on_gpu()
    assert result.shape == (100,)

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_compatibility(backend):
    """Test with multiple backends."""
    result = compute_with_backend(backend)
    assert result is not None

# Configure in pyproject.toml:
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow",
#     "gpu: marks tests requiring GPU",
#     "integration: marks integration tests",
# ]
```

## Best Practices

**DO:**
- Write tests FIRST (TDD)
- Use parametrize for comprehensive coverage
- Test numerical precision with appropriate tolerances
- Test edge cases (empty, NaN, Inf, large arrays)
- Use fixtures for reusable test data
- Use hypothesis for property-based testing
- Benchmark performance-critical code
- Test with multiple array backends

**DON'T:**
- Test implementation details (test behavior)
- Use `time.sleep()` in tests
- Ignore flaky tests (fix or remove them)
- Skip error path testing
- Use exact equality for floating point
- Forget to test array shapes and dtypes

## Integration with CI/CD

```yaml
# GitHub Actions example
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --all-extras
      
      - name: Run tests with coverage
        run: |
          uv run pytest --cov=src --cov-report=xml --cov-report=term-missing
      
      - name: Check coverage threshold
        run: |
          uv run pytest --cov=src --cov-fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

**Remember**: Tests are documentation for scientific code. They demonstrate correctness, numerical precision, and edge case handling. Write them clearly and maintain them diligently.
