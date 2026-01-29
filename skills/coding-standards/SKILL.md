---
name: coding-standards
description: Universal coding standards, best practices, and patterns for Python scientific computing development with NumPy, JAX, and Array API compliance.
---

# Coding Standards & Best Practices

Universal coding standards for Python scientific computing projects.

## Code Quality Principles

### 1. Readability First
- Code is read more than written
- Clear variable and function names
- Self-documenting code with type hints
- Consistent formatting (ruff format)

### 2. KISS (Keep It Simple, Stupid)
- Simplest solution that works
- Avoid over-engineering
- No premature optimization
- Easy to understand > clever code

### 3. DRY (Don't Repeat Yourself)
- Extract common logic into functions
- Create reusable array operations
- Share utilities across modules
- Avoid copy-paste programming

### 4. YAGNI (You Aren't Gonna Need It)
- Don't build features before they're needed
- Avoid speculative generality
- Add complexity only when required
- Start simple, refactor when needed

## Python Scientific Standards

### Variable Naming

```python
# ✅ GOOD: Descriptive names with units
sample_rate_hz = 1000.0
signal_duration_s = 1.0
center_frequency_hz = 5e6
n_elements = 128

# ❌ BAD: Unclear names
sr = 1000
dur = 1.0
fc = 5e6
n = 128
```

### Function Naming

```python
# ✅ GOOD: Verb-noun pattern with type hints
def compute_delays(
    focus: Float[NDArray, "3"],
    elements: Float[NDArray, "n_elem 3"],
    speed: float = 1540.0
) -> Float[NDArray, "n_elem"]:
    """Compute transmit delays for focused wave."""
    pass

def validate_signal_shape(signal: NDArray, expected_ndim: int) -> None:
    """Validate signal has expected dimensionality."""
    pass

def is_valid_frequency(freq: float) -> bool:
    """Check if frequency is in valid range."""
    return 0 < freq < 1e9

# ❌ BAD: Unclear or noun-only
def delays(f, e, s):
    pass

def shape(s, n):
    pass

def frequency(f):
    pass
```

### Immutability Pattern (CRITICAL)

```python
import numpy as np

# ✅ ALWAYS create new arrays
def normalize(signal: NDArray) -> NDArray:
    """Normalize signal (immutable)."""
    centered = signal - signal.mean()
    return centered / centered.std()

# ✅ GOOD: JAX-compatible operations
import jax.numpy as jnp

def normalize_jax(signal: jnp.ndarray) -> jnp.ndarray:
    """JAX automatically enforces immutability."""
    centered = signal - signal.mean()
    return centered / centered.std()

# ❌ NEVER mutate inputs
def normalize_bad(signal: NDArray) -> NDArray:
    signal -= signal.mean()  # Mutates input!
    signal /= signal.std()
    return signal
```

### Type Hints with jaxtyping

```python
from jaxtyping import Float, Int, Complex
from beartype import beartype
from numpy.typing import NDArray

# ✅ GOOD: Complete type hints with shape documentation
@beartype
def beamform(
    signals: Float[NDArray, "channels time"],
    delays: Float[NDArray, "channels"],
    sample_rate: float
) -> Float[NDArray, "time"]:
    """Delay-and-sum beamforming.
    
    Args:
        signals: RF signals, shape (n_channels, n_samples)
        delays: Time delays in seconds, shape (n_channels,)
        sample_rate: Sampling rate in Hz
    
    Returns:
        Beamformed signal, shape (n_samples,)
    """
    pass

# ❌ BAD: No type hints
def beamform(signals, delays, sample_rate):
    pass
```

### Error Handling

```python
import logging
logger = logging.getLogger(__name__)

# ✅ GOOD: Comprehensive error handling with context
def load_signal_data(filepath: Path) -> NDArray:
    """Load signal data from file.
    
    Args:
        filepath: Path to signal file
    
    Returns:
        Signal array
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"Signal file not found: {filepath}")
        
        data = np.load(filepath)
        
        if not isinstance(data, np.ndarray):
            raise ValueError(f"File does not contain array: {filepath}")
        
        if data.size == 0:
            raise ValueError(f"Empty array in file: {filepath}")
        
        return data
        
    except FileNotFoundError:
        raise  # Re-raise file errors
    except Exception as e:
        logger.error(f"Failed to load signal: {filepath}", exc_info=True)
        raise ValueError(f"Failed to load signal data: {e}") from e

# ❌ BAD: Bare except, no context
def load_signal_data_bad(filepath):
    try:
        return np.load(filepath)
    except:
        return None
```

### Array Validation

```python
# ✅ GOOD: Validate inputs early
def process_signal(signal: Float[NDArray, "time"]) -> Float[NDArray, "time"]:
    """Process 1D time-domain signal."""
    # Shape validation
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D")
    
    # Value validation
    if np.any(np.isnan(signal)):
        raise ValueError("Signal contains NaN values")
    
    if np.any(np.isinf(signal)):
        raise ValueError("Signal contains Inf values")
    
    # Size validation
    if signal.size == 0:
        raise ValueError("Signal cannot be empty")
    
    return signal * 2.0

# ❌ BAD: No validation
def process_signal_bad(signal):
    return signal * 2.0
```

## Array API Compliance

### Backend-Agnostic Code

```python
import array_api_compat

# ✅ GOOD: Works with NumPy, JAX, CuPy, PyTorch
def compute_norm(x):
    """Compute L2 norm (backend-agnostic)."""
    xp = array_api_compat.array_namespace(x)
    return xp.sqrt(xp.sum(x ** 2))

# Test with different backends
import numpy as np
import jax.numpy as jnp

np_result = compute_norm(np.array([3.0, 4.0]))
jax_result = compute_norm(jnp.array([3.0, 4.0]))

# ❌ BAD: NumPy-specific
def compute_norm_bad(x):
    return np.sqrt(np.sum(x ** 2))  # Only works with NumPy
```

### Avoid NumPy-Specific Functions

```python
# ❌ BAD: NumPy-specific functions
result = np.asmatrix(array)  # Not in Array API
result = np.asscalar(value)  # Deprecated
result = array[np.newaxis, :]  # Use None instead

# ✅ GOOD: Array API compliant
result = xp.asarray(array)
result = float(value)
result = array[None, :]  # Or xp.expand_dims(array, 0)
```

## Documentation Standards

### Google-Style Docstrings (REQUIRED)

```python
def compute_pressure(
    distance: Float[NDArray, "points"],
    frequency: float,
    speed: float = 1540.0
) -> Float[NDArray, "points"]:
    """Compute acoustic pressure at given distances.
    
    Uses the Rayleigh-Sommerfeld integral for pressure field
    computation in homogeneous media.
    
    Args:
        distance: Distances from source in meters (m), shape (n_points,)
        frequency: Acoustic frequency in Hertz (Hz)
        speed: Speed of sound in m/s (default: 1540 for tissue)
    
    Returns:
        Pressure values in Pascals (Pa), shape (n_points,)
    
    Raises:
        ValueError: If frequency is negative or distance contains NaN
    
    Example:
        >>> distances = np.array([0.01, 0.02, 0.03])
        >>> pressure = compute_pressure(distances, frequency=5e6)
        >>> pressure.shape
        (3,)
    
    References:
        Garcia D. (2022). SIMUS: an open-source simulator for medical
        ultrasound imaging. Computer Methods and Programs in Biomedicine.
    """
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")
    
    if np.any(np.isnan(distance)):
        raise ValueError("Distance array contains NaN")
    
    wavelength = speed / frequency
    k = 2 * np.pi / wavelength
    
    return np.abs(np.exp(1j * k * distance) / distance)
```

### Physical Units in Docstrings (REQUIRED)

```python
# ✅ GOOD: Document units for all physical quantities
def compute_delay(distance: float, speed: float = 1540.0) -> float:
    """Compute time delay for acoustic wave.
    
    Args:
        distance: Path length in meters (m)
        speed: Speed of sound in m/s (default: 1540 for tissue)
    
    Returns:
        Time delay in seconds (s)
    """
    return distance / speed

# ❌ BAD: No units documented
def compute_delay(distance, speed):
    """Compute time delay."""
    return distance / speed
```

## File Organization

### Project Structure

```
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

### File Naming

```
src/my_library/filtering.py      # snake_case for modules
src/my_library/TransducerParams.py  # PascalCase for classes (if separate file)
tests/test_filtering.py           # test_ prefix
```

### Module Size

```python
# ✅ GOOD: Focused modules (200-400 lines typical, 800 max)
# filtering.py - just filtering functions
# transforms.py - just transform functions

# ❌ BAD: Monolithic modules (>1000 lines)
# signal_processing.py - everything in one file
```

## Comments & Documentation

### When to Comment

```python
# ✅ GOOD: Explain WHY, not WHAT
# Use Hanning window to reduce spectral leakage
window = np.hanning(len(signal))

# Deliberately using in-place operation for memory efficiency with large arrays
signal *= window

# ❌ BAD: Stating the obvious
# Multiply signal by 2
signal = signal * 2

# Get length of array
n = len(array)
```

### Inline Comments for Complex Math

```python
def compute_beamforming_delays(
    focus: Float[NDArray, "3"],
    elements: Float[NDArray, "n_elem 3"],
    speed: float = 1540.0
) -> Float[NDArray, "n_elem"]:
    """Compute transmit delays for focused wave.
    
    Args:
        focus: Focus point (x, y, z) in meters
        elements: Element positions (x, y, z) in meters
        speed: Speed of sound in m/s
    
    Returns:
        Delay times in seconds for each element
    """
    # Compute Euclidean distance from each element to focus point
    # Distance = sqrt((x_f - x_e)^2 + (y_f - y_e)^2 + (z_f - z_e)^2)
    distances = np.linalg.norm(elements - focus, axis=1)
    
    # Convert distance to time: t = d / c
    delays = distances / speed
    
    # Subtract minimum delay to start all signals at t=0
    delays -= delays.min()
    
    return delays
```

## Performance Best Practices

### Vectorization

```python
# ✅ GOOD: Vectorized operations
def apply_gain(signals: NDArray, gains: NDArray) -> NDArray:
    """Apply per-channel gains (vectorized)."""
    return signals * gains[:, np.newaxis]

# ❌ BAD: Python loops
def apply_gain_bad(signals, gains):
    result = np.empty_like(signals)
    for i in range(len(signals)):
        result[i] = signals[i] * gains[i]
    return result
```

### Array Preallocation

```python
# ✅ GOOD: Preallocate arrays
def process_batch(signals: list[NDArray]) -> NDArray:
    n = len(signals)
    results = np.empty((n, signals[0].shape[0]))
    for i, signal in enumerate(signals):
        results[i] = process_signal(signal)
    return results

# ❌ BAD: Growing list
def process_batch_bad(signals):
    results = []
    for signal in signals:
        results.append(process_signal(signal))
    return np.array(results)
```

### Avoid Unnecessary Copies

```python
# ✅ GOOD: Use views when possible
def get_subarray(arr: NDArray, start: int, end: int) -> NDArray:
    """Return view of subarray (no copy)."""
    return arr[start:end]

# ❌ BAD: Unnecessary copy
def get_subarray_bad(arr, start, end):
    return arr[start:end].copy()  # Unnecessary if not mutating
```

## Testing Standards

### Test Structure (Arrange-Act-Assert)

```python
class TestComputeDelay:
    def test_computes_correct_delay(self):
        """Compute delay for typical tissue."""
        # Arrange
        distance = 0.03  # 3 cm
        speed = 1540.0   # m/s
        expected = 0.03 / 1540.0
        
        # Act
        result = compute_delay(distance, speed)
        
        # Assert
        assert result == pytest.approx(expected)
```

### Test Naming

```python
# ✅ GOOD: Descriptive test names
def test_returns_empty_when_no_signals():
    pass

def test_raises_value_error_when_frequency_negative():
    pass

def test_preserves_shape_for_2d_arrays():
    pass

# ❌ BAD: Vague test names
def test_works():
    pass

def test_compute():
    pass
```

## Code Smell Detection

### 1. Long Functions

```python
# ❌ BAD: Function > 50 lines
def process_ultrasound_data():
    # 100 lines of code
    pass

# ✅ GOOD: Split into smaller functions
def process_ultrasound_data():
    validated = validate_input_data()
    filtered = apply_bandpass_filter(validated)
    beamformed = apply_beamforming(filtered)
    return envelope_detection(beamformed)
```

### 2. Deep Nesting

```python
# ❌ BAD: 5+ levels of nesting
if signal is not None:
    if signal.size > 0:
        if not np.any(np.isnan(signal)):
            if sample_rate > 0:
                if frequency > 0:
                    # Do something
                    pass

# ✅ GOOD: Early returns
if signal is None:
    raise ValueError("Signal cannot be None")
if signal.size == 0:
    raise ValueError("Signal cannot be empty")
if np.any(np.isnan(signal)):
    raise ValueError("Signal contains NaN")
if sample_rate <= 0:
    raise ValueError("Sample rate must be positive")
if frequency <= 0:
    raise ValueError("Frequency must be positive")

# Do something
```

### 3. Magic Numbers

```python
# ❌ BAD: Unexplained numbers
if frequency > 20000000:
    pass
delay = distance / 1540

# ✅ GOOD: Named constants
MAX_FREQUENCY_HZ = 20e6  # 20 MHz
SPEED_OF_SOUND_TISSUE = 1540.0  # m/s

if frequency > MAX_FREQUENCY_HZ:
    pass
delay = distance / SPEED_OF_SOUND_TISSUE
```

## Anti-Patterns to Avoid

```python
# ❌ BAD: Using import *
from numpy import *

# ✅ GOOD: Explicit imports
import numpy as np

# ❌ BAD: Mutable default arguments
def process(data, buffer=[]):
    buffer.append(data)
    return buffer

# ✅ GOOD: None default with initialization
def process(data: float, buffer: list[float] | None = None) -> list[float]:
    if buffer is None:
        buffer = []
    buffer.append(data)
    return buffer

# ❌ BAD: Bare except
try:
    result = compute()
except:
    pass

# ✅ GOOD: Specific exceptions
try:
    result = compute()
except (ValueError, TypeError) as e:
    logger.error(f"Computation failed: {e}")
    raise

# ❌ BAD: print() in production code
def compute(x):
    print(f"Computing {x}")
    return x ** 2

# ✅ GOOD: Use logging
import logging
logger = logging.getLogger(__name__)

def compute(x: float) -> float:
    logger.debug(f"Computing {x}")
    return x ** 2
```

**Remember**: Code quality is not negotiable. Clear, maintainable code with proper type hints and validation enables rapid development and confident refactoring in scientific computing.
