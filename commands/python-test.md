---
description: Enforce TDD workflow for Python scientific code. Write parametrized pytest tests first, then implement. Verify 80%+ coverage with pytest --cov.
---

# Python TDD Command

This command enforces test-driven development methodology for Python scientific computing using pytest and numerical validation.

## What This Command Does

1. **Define Types/Signatures**: Scaffold function signatures with jaxtyping
2. **Write Parametrized Tests**: Create comprehensive test cases (RED)
3. **Run Tests**: Verify tests fail for the right reason
4. **Implement Code**: Write minimal code to pass (GREEN)
5. **Refactor**: Improve while keeping tests green
6. **Check Coverage**: Ensure 80%+ coverage

## When to Use

Use `/python-test` when:
- Implementing new Python scientific functions
- Adding test coverage to array operations
- Fixing bugs (write failing test first)
- Building numerical algorithms
- Learning TDD workflow for scientific Python

## TDD Cycle

```
RED     → Write failing parametrized test
GREEN   → Implement minimal code to pass
REFACTOR → Improve code, tests stay green
REPEAT  → Next test case
```

## Example Session

```text
User: /python-test I need a function to compute acoustic delays

Agent:
# TDD Session: Acoustic Delay Computation

## Step 1: Define Interface

```python
# src/fast_simus/delays.py
from jaxtyping import Float
from beartype import beartype
from numpy.typing import NDArray

@beartype
def compute_delays(
    focus: Float[NDArray, "3"],
    elements: Float[NDArray, "n_elem 3"],
    speed: float = 1540.0
) -> Float[NDArray, "n_elem"]:
    """Compute transmit delays for focused wave.
    
    Args:
        focus: Focus point coordinates in meters (x, y, z)
        elements: Element positions in meters, shape (n_elements, 3)
        speed: Speed of sound in m/s (default: 1540 for tissue)
    
    Returns:
        Delay times in seconds for each element
    
    Raises:
        ValueError: If speed is non-positive or arrays have NaN
    """
    raise NotImplementedError("not implemented")
```

## Step 2: Write Parametrized Tests (RED)

```python
# tests/test_delays.py
import pytest
import numpy as np
from numpy.testing import assert_allclose
from fast_simus.delays import compute_delays

class TestComputeDelays:
    """Test acoustic delay computation."""
    
    @pytest.mark.parametrize("focus,elements,speed,expected", [
        # Test case 1: Single element at origin
        (
            np.array([0.0, 0.0, 0.03]),  # Focus at 3cm depth
            np.array([[0.0, 0.0, 0.0]]),  # Element at origin
            1540.0,
            np.array([0.03 / 1540.0])  # Expected delay
        ),
        # Test case 2: Two elements symmetric
        (
            np.array([0.0, 0.0, 0.03]),
            np.array([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]),
            1540.0,
            np.array([
                np.sqrt(0.01**2 + 0.03**2) / 1540.0,
                np.sqrt(0.01**2 + 0.03**2) / 1540.0
            ])
        ),
    ])
    def test_computes_correct_delays(self, focus, elements, speed, expected):
        """Compute delays for various configurations."""
        result = compute_delays(focus, elements, speed)
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_rejects_negative_speed(self):
        """Raise error for negative speed."""
        focus = np.array([0.0, 0.0, 0.03])
        elements = np.array([[0.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="speed must be positive"):
            compute_delays(focus, elements, speed=-1540.0)
    
    def test_rejects_nan_in_focus(self):
        """Raise error for NaN in focus."""
        focus = np.array([0.0, np.nan, 0.03])
        elements = np.array([[0.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="contains NaN"):
            compute_delays(focus, elements)
    
    def test_preserves_element_count(self):
        """Output has same length as number of elements."""
        focus = np.array([0.0, 0.0, 0.03])
        elements = np.random.randn(128, 3)  # 128 elements
        
        result = compute_delays(focus, elements)
        assert result.shape == (128,)
```

## Step 3: Run Tests - Verify FAIL

```bash
$ pytest tests/test_delays.py -v

tests/test_delays.py::TestComputeDelays::test_computes_correct_delays[focus0-elements0-1540.0-expected0] FAILED
    NotImplementedError: not implemented

FAILED tests/test_delays.py
```

✓ Tests fail as expected (NotImplementedError).

## Step 4: Implement Minimal Code (GREEN)

```python
# src/fast_simus/delays.py
import numpy as np
from jaxtyping import Float
from beartype import beartype
from numpy.typing import NDArray

@beartype
def compute_delays(
    focus: Float[NDArray, "3"],
    elements: Float[NDArray, "n_elem 3"],
    speed: float = 1540.0
) -> Float[NDArray, "n_elem"]:
    """Compute transmit delays for focused wave.
    
    Args:
        focus: Focus point coordinates in meters (x, y, z)
        elements: Element positions in meters, shape (n_elements, 3)
        speed: Speed of sound in m/s (default: 1540 for tissue)
    
    Returns:
        Delay times in seconds for each element
    
    Raises:
        ValueError: If speed is non-positive or arrays have NaN
    """
    # Validation
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    
    if np.any(np.isnan(focus)):
        raise ValueError("focus contains NaN")
    
    if np.any(np.isnan(elements)):
        raise ValueError("elements contains NaN")
    
    # Compute Euclidean distance from each element to focus
    distances = np.linalg.norm(elements - focus, axis=1)
    
    # Convert distance to time: t = d / c
    delays = distances / speed
    
    # Normalize to start at t=0
    delays -= delays.min()
    
    return delays
```

## Step 5: Run Tests - Verify PASS

```bash
$ pytest tests/test_delays.py -v

tests/test_delays.py::TestComputeDelays::test_computes_correct_delays[...] PASSED
tests/test_delays.py::TestComputeDelays::test_rejects_negative_speed PASSED
tests/test_delays.py::TestComputeDelays::test_rejects_nan_in_focus PASSED
tests/test_delays.py::TestComputeDelays::test_preserves_element_count PASSED

PASSED (4 tests)
```

✓ All tests passing!

## Step 6: Check Coverage

```bash
$ pytest --cov=src/fast_simus --cov-report=term-missing tests/test_delays.py

---------- coverage: platform darwin, python 3.11 -----------
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/fast_simus/delays.py         15      0   100%
-----------------------------------------------------------
TOTAL                            15      0   100%
```

✓ Coverage: 100%

## TDD Complete!
```

## Test Patterns

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (np.array([1, 2, 3]), 6),
    (np.array([0, 0, 0]), 0),
    (np.array([-1, -2, -3]), -6),
])
def test_sum_array(input, expected):
    assert np.sum(input) == expected
```

### Fixtures for Reusable Data
```python
@pytest.fixture
def sample_signal():
    """Generate sample signal."""
    return np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))

def test_filter(sample_signal):
    result = filter_signal(sample_signal)
    assert result.shape == sample_signal.shape
```

### Property-Based Tests
```python
from hypothesis import given
from hypothesis.extra.numpy import arrays

@given(arrays(dtype=np.float64, shape=(100,)))
def test_normalize_properties(signal):
    normalized = normalize(signal)
    assert abs(normalized.mean()) < 1e-10
    assert abs(normalized.std() - 1.0) < 1e-10
```

## Coverage Commands

```bash
# Basic coverage
pytest --cov=src

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html

# Fail if below threshold
pytest --cov=src --cov-fail-under=80

# Coverage for specific module
pytest --cov=src.filtering tests/test_filtering.py
```

## Coverage Targets

| Code Type | Target |
|-----------|--------|
| Core algorithms | 95%+ |
| Public APIs | 90%+ |
| Utilities | 80%+ |
| Generated code | Exclude |

## TDD Best Practices

**DO:**
- Write test FIRST, before any implementation
- Run tests after each change
- Use parametrize for comprehensive coverage
- Test numerical precision with appropriate tolerances
- Include edge cases (empty, NaN, Inf, large arrays)

**DON'T:**
- Write implementation before tests
- Skip the RED phase
- Use exact equality for floating point
- Ignore flaky tests
- Test implementation details

## Related Commands

- `/python-build` - Fix type/lint errors
- `/python-review` - Review code after implementation
- `/verify` - Run full verification loop

## Related

- Skill: `skills/python-scientific-testing/`
- Skill: `skills/tdd-workflow/`
