---
name: python-scientific-patterns
description: Idiomatic Python scientific computing patterns, Array API compliance, and best practices for building robust, efficient NumPy/JAX/CuPy applications.
---

# Python Scientific Computing Patterns

Idiomatic Python patterns and best practices for building robust, efficient, and maintainable scientific computing applications with Array API compliance.

## When to Activate

- Writing new Python scientific code
- Reviewing Python numerical computing code
- Refactoring existing scientific Python code
- Designing array computation libraries

## Core Principles

### 1. Array API Compliance

All numerical operations should use the Array API Standard for backend portability.

```python
# Good: Array API compliant
import array_api_compat
import numpy as np

def compute_norm(x):
    xp = array_api_compat.array_namespace(x)
    return xp.sqrt(xp.sum(x ** 2))

# Works with NumPy, JAX, CuPy, PyTorch

# Bad: NumPy-specific
def compute_norm(x):
    return np.sqrt(np.sum(x ** 2))  # Only works with NumPy
```

### 2. Immutable Array Operations

Prefer functional style that creates new arrays rather than mutating inputs.

```python
# Good: Immutable, works with JAX
def normalize(signal: NDArray) -> NDArray:
    centered = signal - signal.mean()
    return centered / centered.std()

# Bad: Mutates input, breaks JAX
def normalize(signal: NDArray) -> NDArray:
    signal -= signal.mean()  # In-place mutation!
    signal /= signal.std()
    return signal
```

### 3. Type Safety with jaxtyping

Document array shapes and types for self-documenting, validated code.

```python
from typing import Any
from jaxtyping import Float, Int, Num, jaxtyped
from beartype import beartype as typechecker
import array_api_compat
from numpy.typing import NDArray

# Type alias until Array Protocol is standardized
ArrayAPIObj = Any

@jaxtyped(typechecker=typechecker)
def beamform(
    signals: Float[ArrayAPIObj, "channels time"],
    delays: Float[ArrayAPIObj, "channels"]
) -> Float[ArrayAPIObj, "time"]:
    """Delay-and-sum beamforming with shape validation."""
    xp = array_api_compat.array_namespace(signals)
    # Implementation with runtime shape checking
    return xp.sum(signals, axis=0)
```

### Shape Annotation Patterns

```python
# Fixed dimensions
Float[ArrayAPIObj, "height width"]           # 2D image
Float[ArrayAPIObj, "n_samples n_elements"]   # RF signals

# Variable/batch dimensions
Float[ArrayAPIObj, "*batch n_freq"]          # Any batch dims + frequency
Num[ArrayAPIObj, "..."]                      # Any shape

# Named dimensions (self-documenting)
Float[ArrayAPIObj, "n_scatterers"]           # 1D array of scatterer coords
```

### Backend-Specific Optimizations

For operations not in Array API or needing backend-specific speedups:

```python
from array_api_compat import is_jax_array, is_cupy_array, is_numpy_array

def frequency_loop(exp_init, exp_df, n_freq):
    """Frequency loop with backend-specific optimizations."""
    xp = array_api_compat.array_namespace(exp_init)

    if is_jax_array(exp_init):
        # Use JAX lax.scan for efficient accumulation
        import jax.lax as lax
        return _jax_scan_loop(exp_init, exp_df, n_freq)

    elif is_cupy_array(exp_init):
        # Use CuPy kernel fusion
        import cupy
        return _cupy_fused_loop(exp_init, exp_df, n_freq)

    else:
        # Pure Array API fallback (works for NumPy, array-api-strict)
        return _array_api_loop(exp_init, exp_df, n_freq)
```

### Helper Functions for Missing Array API Features

Some functions aren't in the standard yet. Implement with backend branches:

```python
def histogram(x, bins, range=None, weights=None, density=False):
    """Array-api compatible histogram."""
    xp = array_api_compat.array_namespace(x)

    if is_numpy_array(x):
        import numpy as np
        return np.histogram(x, bins=bins, range=range, weights=weights, density=density)

    elif is_jax_array(x):
        import jax.numpy as jnp
        return jnp.histogram(x, bins=bins, range=range, weights=weights, density=density)

    else:
        # Fallback or warn
        import warnings
        warnings.warn(f"histogram not optimized for {xp.__name__}")
        import numpy as np
        return np.histogram(np.asarray(x), bins=bins, range=range, weights=weights, density=density)
```

### Complete Array Function Signature Pattern

Full example with Array API compliance, validation, and documentation:

```python
from typing import Any
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
import array_api_compat

ArrayAPIObj = Any

@jaxtyped(typechecker=typechecker)
def compute_delays(
    focus: Float[ArrayAPIObj, "3"],
    elements: Float[ArrayAPIObj, "n_elem 3"],
    speed: float = 1540.0
) -> Float[ArrayAPIObj, "n_elem"]:
    """Compute transmit delays for focused wave.
    
    Args:
        focus: Focus point coordinates in meters (x, y, z)
        elements: Element positions in meters, shape (n_elements, 3)
        speed: Speed of sound in m/s (default: 1540 for tissue)
        
    Returns:
        Delay times in seconds for each element
    
    Raises:
        ValueError: If speed is non-positive or arrays contain NaN
    
    Example:
        >>> import numpy as np
        >>> focus = np.array([0.0, 0.0, 0.03])
        >>> elements = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        >>> delays = compute_delays(focus, elements)
        >>> delays.shape
        (2,)
    """
    # Get array namespace for backend portability
    xp = array_api_compat.array_namespace(focus, elements)
    
    # Validation
    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    if xp.any(xp.isnan(focus)) or xp.any(xp.isnan(elements)):
        raise ValueError("Arrays contain NaN values")
    
    # Computation using Array API
    # Compute Euclidean distance from each element to focus point
    diff = elements - focus
    distances = xp.sqrt(xp.sum(diff ** 2, axis=1))
    
    # Convert distance to time: t = d / c
    delays = distances / speed
    
    # Normalize to start at t=0 (immutable operation)
    min_delay = xp.min(delays)
    delays = delays - min_delay
    
    return delays
```

## Error Handling Patterns

### Exception Chaining with Context

```python
# Good: Chain exceptions with context
def load_signal_data(path: Path) -> NDArray:
    try:
        data = np.load(path)
    except FileNotFoundError as e:
        raise ValueError(f"Signal file not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load signal: {path}") from e
    
    if data.size == 0:
        raise ValueError(f"Empty signal file: {path}")
    
    return data
```

### Custom Exception Types

```python
# Define domain-specific exceptions
class ArrayShapeError(ValueError):
    """Raised when array shape doesn't match expected dimensions."""
    
    def __init__(self, expected: tuple, actual: tuple, name: str = "array"):
        self.expected = expected
        self.actual = actual
        self.name = name
        super().__init__(
            f"{name} shape mismatch: expected {expected}, got {actual}"
        )

class NumericalError(RuntimeError):
    """Raised when numerical computation produces invalid results."""
    pass

# Sentinel errors for common cases
class ArrayValidationError(ValueError):
    """Base class for array validation errors."""
    pass
```

### Never Ignore Errors

```python
# Bad: Silently catching exceptions
try:
    result = compute_fft(signal)
except Exception:
    result = None  # Silent failure!

# Good: Handle or log and re-raise
import logging
logger = logging.getLogger(__name__)

try:
    result = compute_fft(signal)
except ValueError as e:
    logger.error(f"FFT computation failed: {e}", exc_info=True)
    raise
```

## Array Validation Patterns

### Shape Validation

```python
def validate_signal_shape(
    signal: NDArray,
    expected_ndim: int = 1
) -> None:
    """Validate signal has expected dimensionality."""
    if signal.ndim != expected_ndim:
        raise ArrayShapeError(
            expected=(f"{expected_ndim}D",),
            actual=(f"{signal.ndim}D",),
            name="signal"
        )

@beartype
def process_signal(
    signal: Float[NDArray, "time"]
) -> Float[NDArray, "time"]:
    """Process 1D signal with automatic shape validation."""
    validate_signal_shape(signal, expected_ndim=1)
    
    # Check for invalid values using Array API
    xp = array_api_compat.array_namespace(signal)
    if xp.any(xp.isnan(signal)) or xp.any(xp.isinf(signal)):
        raise NumericalError("Signal contains NaN or Inf values")
    
    return signal * 2.0
```

### Dtype Validation

```python
def ensure_float64(array: NDArray) -> NDArray:
    """Ensure array is float64, converting if necessary."""
    if array.dtype != np.float64:
        logger.warning(f"Converting {array.dtype} to float64")
        return array.astype(np.float64)
    return array

def validate_compatible_dtypes(*arrays: NDArray) -> None:
    """Validate all arrays have compatible dtypes."""
    dtypes = [arr.dtype for arr in arrays]
    if len(set(dtypes)) > 1:
        raise TypeError(f"Incompatible dtypes: {dtypes}")
```

## Parallel Computation Patterns

### JAX Parallelization

```python
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap

# Vectorization with vmap
@jit
def process_single(signal: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft(signal)

# Automatically vectorize over batch dimension
process_batch = vmap(process_single)

# Multi-device parallelization
@pmap
def process_parallel(signals: jnp.ndarray) -> jnp.ndarray:
    """Process signals in parallel across devices."""
    return jnp.fft.fft(signals)

# Usage
signals = jnp.array([...])  # Shape: (n_devices, n_samples)
results = process_parallel(signals)
```

### concurrent.futures for CPU Parallelism

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

def process_file(filepath: Path) -> NDArray:
    """Process single file."""
    data = np.load(filepath)
    return compute_features(data)

def process_files_parallel(
    filepaths: List[Path],
    max_workers: int = 4
) -> List[NDArray]:
    """Process multiple files in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_file, path): path
            for path in filepaths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                raise
    
    return results
```

## Protocol Design (Python's Interfaces)

### Small, Focused Protocols

```python
from typing import Protocol, runtime_checkable
from numpy.typing import NDArray

@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array-like objects."""
    
    def __array__(self) -> NDArray:
        """Convert to NumPy array."""
        ...

@runtime_checkable
class Transformable(Protocol):
    """Protocol for objects that can be transformed."""
    
    def transform(self, matrix: NDArray) -> NDArray:
        """Apply transformation matrix."""
        ...

# Use protocols for flexible typing
def process_array_like(data: ArrayLike) -> NDArray:
    """Accept any array-like object."""
    arr = np.asarray(data)
    return arr * 2
```

### Abstract Base Classes for Algorithms

```python
from abc import ABC, abstractmethod

class SignalProcessor(ABC):
    """Base class for signal processing algorithms."""
    
    @abstractmethod
    def process(self, signal: NDArray) -> NDArray:
        """Process signal and return result."""
        pass
    
    def validate_input(self, signal: NDArray) -> None:
        """Validate input signal (common logic)."""
        xp = array_api_compat.array_namespace(signal)
        if signal.size == 0:
            raise ValueError("Empty signal")
        if xp.any(xp.isnan(signal)):
            raise ValueError("Signal contains NaN")

class BandpassFilter(SignalProcessor):
    """Bandpass filter implementation."""
    
    def __init__(self, lowcut: float, highcut: float, fs: float):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
    
    def process(self, signal: NDArray) -> NDArray:
        self.validate_input(signal)
        # Implementation
        return filtered_signal
```

## Package Organization

### Standard Scientific Python Layout

```text
my_array_library/
├── src/
│   └── my_library/
│       ├── __init__.py
│       ├── py.typed              # PEP 561 marker
│       ├── core/
│       │   ├── __init__.py
│       │   ├── arrays.py         # Array utilities
│       │   └── validation.py    # Input validation
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── filtering.py     # Signal filtering
│       │   └── transforms.py    # FFT, wavelets
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── numpy_backend.py
│       │   └── jax_backend.py
│       └── utils/
│           ├── __init__.py
│           ├── constants.py     # Physical constants
│           └── types.py         # Type definitions
├── tests/
│   ├── test_arrays.py
│   ├── test_filtering.py
│   └── conftest.py              # pytest fixtures
├── pyproject.toml
├── README.md
└── LICENSE
```

### Module Naming

```python
# Good: Short, lowercase, descriptive
import numpy as np
import jax.numpy as jnp
from my_library import filtering
from my_library.algorithms import beamforming

# Bad: Verbose or unclear
from my_library.algorithms.signal_processing_filters import bandpass
```

### Avoid Module-Level State

```python
# Bad: Global mutable state
_cache = {}

def compute(x):
    if x in _cache:
        return _cache[x]
    result = expensive_computation(x)
    _cache[x] = result
    return result

# Good: Explicit state management
from functools import lru_cache

@lru_cache(maxsize=128)
def compute(x: float) -> float:
    """Cached computation with explicit size limit."""
    return expensive_computation(x)

# Or use a class
class ComputationCache:
    def __init__(self, maxsize: int = 128):
        self._cache: dict = {}
        self._maxsize = maxsize
    
    def compute(self, x: float) -> float:
        if x in self._cache:
            return self._cache[x]
        result = expensive_computation(x)
        if len(self._cache) < self._maxsize:
            self._cache[x] = result
        return result
```

## Dataclass Patterns

### Configuration with dataclasses

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)  # Immutable
class TransducerParams:
    """Ultrasound transducer parameters."""
    
    fc: float  # Center frequency (Hz)
    pitch: float  # Element pitch (m)
    n_elements: int  # Number of elements
    width: float  # Element width (m)
    
    # Optional with defaults
    height: float = float('inf')  # Element height (m)
    focus: float = float('inf')  # Elevation focus (m)
    radius: float = float('inf')  # Curvature radius (m)
    bandwidth: float = 0.75  # Fractional bandwidth
    c: float = 1540.0  # Speed of sound (m/s)
    fs: float = field(init=False)  # Computed field
    
    def __post_init__(self):
        # Validation
        if self.fc <= 0:
            raise ValueError("Center frequency must be positive")
        if self.n_elements <= 0:
            raise ValueError("Number of elements must be positive")
        
        # Computed field
        object.__setattr__(self, 'fs', 4 * self.fc)

# Factory functions for common configurations
def L11_5v() -> TransducerParams:
    """Verasonics L11-5v linear array."""
    return TransducerParams(
        fc=7.6e6,
        pitch=0.3e-3,
        n_elements=128,
        width=0.27e-3,
        height=5e-3
    )
```

### Options Pattern with dataclasses

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ComputeOptions:
    """Options for array computation."""
    
    backend: Literal["numpy", "jax", "cupy"] = "numpy"
    dtype: np.dtype = np.float64
    device: Optional[str] = None  # "cpu", "cuda:0", etc.
    validate_input: bool = True
    check_nan: bool = True
    
    def __post_init__(self):
        if self.backend == "jax" and self.device is None:
            self.device = "cpu"

def compute_with_options(
    data: NDArray,
    options: Optional[ComputeOptions] = None
) -> NDArray:
    """Compute with configurable options."""
    if options is None:
        options = ComputeOptions()
    
    if options.validate_input:
        validate_array(data)
    
    # Use options...
    return result
```

## Memory and Performance

### Preallocate Arrays When Size is Known

```python
# Bad: Growing list then converting
def process_signals(signals: List[NDArray]) -> NDArray:
    results = []
    for signal in signals:
        results.append(process(signal))
    return np.array(results)  # Slow conversion

# Good: Preallocate array
def process_signals(signals: List[NDArray]) -> NDArray:
    n = len(signals)
    results = np.empty((n, signals[0].shape[0]))
    for i, signal in enumerate(signals):
        results[i] = process(signal)
    return results
```

### Use Array Views Instead of Copies

```python
# Bad: Unnecessary copy
def get_subarray(arr: NDArray, start: int, end: int) -> NDArray:
    return arr[start:end].copy()  # Unnecessary copy

# Good: Return view (if mutation is not a concern)
def get_subarray(arr: NDArray, start: int, end: int) -> NDArray:
    return arr[start:end]  # View, no copy

# If mutation is a concern, document it
def get_subarray_copy(arr: NDArray, start: int, end: int) -> NDArray:
    """Return copy of subarray to prevent mutation."""
    return arr[start:end].copy()
```

### Vectorize Operations

```python
# Bad: Python loop
def apply_gain(signals: NDArray, gains: NDArray) -> NDArray:
    result = np.empty_like(signals)
    for i in range(len(signals)):
        result[i] = signals[i] * gains[i]
    return result

# Good: Vectorized
def apply_gain(signals: NDArray, gains: NDArray) -> NDArray:
    return signals * gains[:, np.newaxis]  # Broadcasting
```

## Testing Array API Compliance

Use `array-api-strict` to verify Array API compliance:

```python
import pytest
import numpy
import array_api_strict

@pytest.mark.parametrize("xp", [numpy, array_api_strict])
def test_delays_array_api_compliant(xp):
    """Test works with strict Array API implementation."""
    positions = xp.linspace(-0.01, 0.01, 64)
    delays = compute_delays(positions, focus_x=0.0, focus_z=0.03)

    assert delays.shape == positions.shape
    assert hasattr(delays, '__array_namespace__')  # Is Array API compliant
```

### Key Rules for Array API Compliance

1. **Always use `xp = array_namespace(input)`** - never hardcode numpy
2. **Preserve input types** - output arrays should match input backend
3. **Test with array-api-strict** - catches non-compliant operations
4. **Document shapes** - use jaxtyping annotations on all public functions
5. **Branch for optimization** - use `is_jax_array()` etc. for backend-specific code

## Python Tooling Integration

### Essential Commands

```bash
# Type checking
ty check src/

# Linting and formatting
ruff check .
ruff format .

# Testing
pytest
pytest --cov=src --cov-report=term-missing
pytest --benchmark-only

# Dependency management
uv sync
uv add numpy
uv add --dev pytest
```

### Recommended Configuration (pyproject.toml)

```toml
[project]
name = "my-array-library"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "array-api-compat>=1.4",
    "jaxtyping>=0.2.28",
    "beartype>=0.18.0",
]

[project.optional-dependencies]
jax = ["jax[cpu]>=0.4.0"]
cupy = ["cupy>=12.0.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-benchmark>=4.0.0",
    "ruff>=0.5.0",
    "array-api-strict>=2.0",  # For Array API compliance testing
]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "ANN", # flake8-annotations
    "B",   # flake8-bugbear
    "NPY", # NumPy-specific rules
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--strict-markers"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*"]

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

## Quick Reference: Python Scientific Idioms

| Idiom | Description |
|-------|-------------|
| Array API compliance | Use array namespace for backend portability |
| Immutable operations | Create new arrays, don't mutate inputs |
| Type hints with jaxtyping | Document array shapes in function signatures |
| Validate early | Check shapes, dtypes, NaN/Inf at function entry |
| Vectorize operations | Avoid Python loops, use array operations |
| Use protocols | Define interfaces with typing.Protocol |
| Functional style | Pure functions without side effects |
| Explicit is better than implicit | Clear validation over silent assumptions |

## Anti-Patterns to Avoid

```python
# Bad: In-place mutation (breaks JAX)
def normalize(arr):
    arr -= arr.mean()
    arr /= arr.std()
    return arr

# Bad: NumPy-specific functions
result = np.asmatrix(array)  # Not in Array API

# Bad: Mutable default arguments
def process(data, buffer=[]):  # Dangerous!
    buffer.append(data)
    return buffer

# Bad: Using print instead of logging
def compute(x):
    print(f"Computing {x}")  # Use logger.info()
    return x ** 2

# Bad: Bare except
try:
    result = compute()
except:  # Too broad!
    pass

# Bad: Not using type hints
def process(signal):  # What type? What shape?
    return signal * 2

# Good: Type hints with shape documentation
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

@jaxtyped(typechecker=typechecker)
def process(signal: Float[ArrayAPIObj, "time"]) -> Float[ArrayAPIObj, "time"]:
    """Process 1D time-domain signal."""
    return signal * 2
```

**Remember**: Scientific Python code should be correct, fast, and maintainable. When in doubt, prioritize correctness and clarity over premature optimization.
