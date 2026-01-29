---
description: Comprehensive Python scientific code review for Array API compliance, type safety, numerical precision, and performance. Invokes the python-reviewer agent.
---

# Python Scientific Code Review

This command invokes the **python-reviewer** agent for comprehensive Python scientific computing code review.

## What This Command Does

1. **Identify Python Changes**: Find modified `.py` files via `git diff`
2. **Run Static Analysis**: Execute `ruff check`, `ty check`
3. **Array Validation Check**: Verify shape/dtype validation, NaN/Inf handling
4. **Type Safety Review**: Check jaxtyping annotations, beartype usage
5. **Array API Compliance**: Verify backend-agnostic code
6. **Generate Report**: Categorize issues by severity

## When to Use

Use `/python-review` when:
- After writing or modifying Python scientific code
- Before committing Python changes
- Reviewing pull requests with array operations
- Onboarding to Array API patterns
- Learning scientific Python best practices

## Review Categories

### CRITICAL (Must Fix)
- Missing array shape/dtype validation
- NaN/Inf propagation without checks
- Path traversal vulnerabilities
- Hardcoded credentials
- Mutable default arguments
- In-place array mutations (breaks JAX)

### HIGH (Should Fix)
- Missing type hints (jaxtyping)
- Missing beartype decorators
- NumPy-specific functions (not Array API)
- Missing docstrings (Google style)
- Bare except clauses
- print() instead of logging

### MEDIUM (Consider)
- Non-idiomatic Python patterns
- Inefficient array operations (loops vs vectorization)
- Missing physical units in docstrings
- Array copies when views would suffice
- Magic numbers without constants

## Automated Checks Run

```bash
# Type checking
ty check src/

# Linting
ruff check .

# Test suite
pytest

# Coverage
pytest --cov=src --cov-report=term-missing
```

## Example Usage

```text
User: /python-review

Agent:
# Python Scientific Code Review Report

## Files Reviewed
- src/fast_simus/pfield.py (modified)
- src/fast_simus/validation.py (modified)

## Static Analysis Results
✓ ruff: No issues
✓ ty: No issues

## Issues Found

[CRITICAL] Missing Array Validation
File: src/fast_simus/pfield.py:42
Issue: Array used without shape/dtype validation
```python
def compute_pfield(positions, frequency):
    return positions * frequency  # No validation!
```
Fix: Add validation and type hints
```python
from jaxtyping import Float
from beartype import beartype

@beartype
def compute_pfield(
    positions: Float[NDArray, "points 3"],
    frequency: float
) -> Float[NDArray, "points"]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) positions, got {positions.shape}")
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")
    
    return np.linalg.norm(positions, axis=1) * frequency
```

[HIGH] NumPy-Specific Function
File: src/fast_simus/transforms.py:15
Issue: Using NumPy-specific function (not Array API)
```python
result = np.asmatrix(array)  # Not in Array API
```
Fix: Use Array API compliant function
```python
import array_api_compat
xp = array_api_compat.array_namespace(array)
result = xp.asarray(array)
```

[HIGH] Missing Docstring
File: src/fast_simus/validation.py:8
Issue: Public function missing Google-style docstring
```python
def validate_signal(signal):
    pass
```
Fix: Add comprehensive docstring
```python
def validate_signal(signal: Float[NDArray, "time"]) -> None:
    """Validate 1D time-domain signal.
    
    Args:
        signal: Time-domain signal array
    
    Raises:
        ValueError: If signal is not 1D or contains NaN/Inf
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or Inf")
```

## Summary
- CRITICAL: 1
- HIGH: 2
- MEDIUM: 0

Recommendation: ❌ Block merge until CRITICAL and HIGH issues are fixed
```

## Approval Criteria

| Status | Condition |
|--------|-----------|
| ✅ Approve | No CRITICAL or HIGH issues |
| ⚠️ Warning | Only MEDIUM issues (merge with caution) |
| ❌ Block | CRITICAL or HIGH issues found |

## Integration with Other Commands

- Use `/python-test` first to ensure tests pass
- Use `/python-build` if type/lint errors occur
- Use `/python-review` before committing
- Use `/code-review` for non-Python specific concerns

## Related

- Agent: `agents/python-reviewer.md`
- Skills: `skills/python-scientific-patterns/`, `skills/python-scientific-testing/`
