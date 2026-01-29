---
name: googledoc-docstrings
description: Guide for writing Google-style docstrings in Python. Covers functions, classes, modules, and integration with type hints.
---

# Google-style Docstrings

Comprehensive guide for writing [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) in Python projects.

## When to Activate

- Writing new Python functions, classes, or modules
- Documenting public APIs
- Project uses Google-style docstrings (check existing code)
- User asks about docstring format or documentation

## Function Docstring Template

```python
def compute_delays(
    focus: Float[NDArray, "3"],
    elements: Float[NDArray, "n_elem 3"],
    speed: float = 1540.0,
) -> Float[NDArray, "n_elem"]:
    """Compute transmit delays for focused wave.

    Uses Euclidean distance from each element to focus point,
    then converts to time delays.

    Args:
        focus: Focus point coordinates in meters (x, y, z).
        elements: Element positions in meters, shape (n_elements, 3).
        speed: Speed of sound in m/s. Defaults to 1540 (soft tissue).

    Returns:
        Delay times in seconds for each element. Normalized so minimum
        delay is zero.

    Raises:
        ValueError: If speed is non-positive.
        ValueError: If arrays contain NaN values.

    Note:
        Assumes far-field approximation where elements are small
        relative to the propagation distance.

    Example:
        >>> import numpy as np
        >>> focus = np.array([0.0, 0.0, 0.03])
        >>> elements = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        >>> delays = compute_delays(focus, elements)
        >>> delays.shape
        (2,)

    References:
        Smith J. Acoustic Beamforming Fundamentals. 2020.
        https://doi.org/10.xxxx/example
    """
```

## Section Reference

### Args

Document each parameter. Type hints are in signature, so focus on description:

```python
Args:
    param1: Description of first parameter. Can span multiple lines
        with proper indentation (4 spaces continuation).
    param2: Second parameter. Defaults to None.
    *args: Variable positional arguments for batch processing.
    **kwargs: Additional keyword arguments passed to underlying function.
```

### Returns

Describe the return value(s):

```python
# Single return
Returns:
    Description of return value. Include shape for arrays.

# Multiple returns (tuple)
Returns:
    Tuple containing:
    - result: The computed result with shape (n, m).
    - metadata: Dictionary with computation statistics.

# Named tuple
Returns:
    ProcessResult with fields:
    - data: Processed array.
    - status: Success/failure indicator.
```

### Raises

List exceptions that may be raised:

```python
Raises:
    ValueError: If input arrays have incompatible shapes.
    TypeError: If `config` is not a dict or Config instance.
    FileNotFoundError: If the specified path does not exist.
```

### Note

Additional information, caveats, assumptions:

```python
Note:
    This function assumes single-precision (float32) inputs for
    performance. Double precision is supported but slower.

    Memory usage scales as O(n * m) where n is the number of
    points and m is the number of frequencies.
```

### Example

Executable examples (run with `pytest --doctest-modules`):

```python
Example:
    >>> import numpy as np
    >>> signal = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> filtered = lowpass_filter(signal, cutoff=10, fs=100)
    >>> filtered.shape
    (100,)

    Multiple examples can be separated by blank lines:

    >>> normalize(np.array([1, 2, 3]))
    array([-1.22...,  0.        ,  1.22...])
```

### References

Citations and links:

```python
References:
    Garcia D. SIMUS: an open-source simulator. CMPB, 2022.
    https://doi.org/10.1016/j.cmpb.2022.106726

    Smith J, Doe A. Algorithm improvements. Journal, 2023.
```

## Class Docstring Template

```python
@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for signal processing pipeline.

    Immutable dataclass containing all parameters for the processing
    pipeline. Use factory functions like `default_config()` for common
    configurations.

    Attributes:
        sample_rate: Sampling frequency in Hz.
        window_size: Analysis window size in samples.
        overlap: Overlap between windows as fraction (0.0-1.0).
        normalize: Whether to normalize output. Defaults to True.

    Example:
        >>> config = ProcessingConfig(sample_rate=1000, window_size=256)
        >>> config.sample_rate
        1000
    """

    sample_rate: float
    window_size: int
    overlap: float = 0.5
    normalize: bool = True
```

## Module Docstring

At top of file, before imports:

```python
"""Signal processing utilities for time-frequency analysis.

This module provides functions for computing spectrograms, wavelets,
and other time-frequency representations of signals.

The main functions are:
- `compute_spectrogram()`: Short-time Fourier transform
- `compute_wavelet()`: Continuous wavelet transform
- `compute_hilbert()`: Hilbert transform for envelope

Example:
    >>> from mypackage import signal_processing
    >>> spectrogram = signal_processing.compute_spectrogram(data, fs=1000)
"""

import numpy as np
# ... rest of imports
```

## Type Hints + Docstrings

With type hints in signature, docstring describes semantics (not types):

```python
from jaxtyping import Float
from beartype import beartype
from numpy.typing import NDArray

@beartype
def bandpass_filter(
    signal: Float[NDArray, "time"],
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> Float[NDArray, "time"]:
    """Apply Butterworth bandpass filter to signal.

    Args:
        signal: Input signal to filter.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order. Higher = sharper cutoff but more ringing.

    Returns:
        Filtered signal with same shape as input.
    """
```

## Doctest Validation

Run doctests to verify examples work:

```bash
# Run doctests for a module
pytest --doctest-modules src/mypackage/

# Run doctests for specific file
pytest --doctest-modules src/mypackage/utils.py

# Include doctests in regular test run
pytest --doctest-modules
```

## Common Mistakes

### Don't Repeat Type Hints in Args

```python
# ❌ Wrong - type is already in signature
Args:
    x (np.ndarray): The input array.
    n (int): Number of iterations.

# ✅ Correct - focus on description
Args:
    x: Input array to process.
    n: Number of iterations. Higher values = more accurate.
```

### Don't Use NumPy-style Section Headers

```python
# ❌ Wrong - NumPy style
Parameters
----------
x : array
    The input.

# ✅ Correct - Google style
Args:
    x: The input array.
```

### Don't Skip the Summary Line

```python
# ❌ Wrong - no summary
def process(x):
    """
    Args:
        x: Input data.
    """

# ✅ Correct - summary first
def process(x):
    """Process input data and return result.

    Args:
        x: Input data.
    """
```

### Don't Document Private Methods Extensively

```python
# ✅ Private method - minimal docstring is fine
def _validate_input(self, x):
    """Validate input array."""
    ...

# ✅ Public method - full docstring
def process(self, x):
    """Process input array.

    Args:
        x: Input array with shape (n, m).

    Returns:
        Processed array.
    """
```

## Integration with Documentation Tools

### Sphinx with Napoleon

```python
# conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
```

### MkDocs with mkdocstrings

```yaml
# mkdocs.yml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
```

## Best Practices

1. **Summary line first** - One line describing what the function does
2. **Args describe semantics** - Not types (those are in signature)
3. **Include units** - "distance in meters", "frequency in Hz"
4. **Document defaults** - "Defaults to 1540 (soft tissue)"
5. **Mention shape** - For arrays, describe expected dimensions
6. **Add examples** - Executable examples catch documentation rot
7. **Keep it concise** - More detail in Note section if needed
