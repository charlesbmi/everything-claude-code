# Testing Requirements

## Minimum Test Coverage: 80%

Test Types (ALL required):
1. **Unit Tests** - Individual functions, array operations, algorithms
2. **Integration Tests** - Multi-step pipelines, file I/O, backend compatibility
3. **Numerical Validation** - Precision tests against reference implementations

## Test-Driven Development

MANDATORY workflow:
1. Write test first (RED)
2. Run test - it should FAIL
3. Write minimal implementation (GREEN)
4. Run test - it should PASS
5. Refactor (IMPROVE)
6. Verify coverage (80%+)

## Scientific Testing Requirements

### Numerical Precision
- Use `np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-8)` or array-api equivalent for floating point comparison
- Test against reference implementations (PyMUST)

### Edge Cases
- Empty arrays: `np.array([])`
- NaN values: `np.array([1, np.nan, 3])`
- Inf values: `np.array([1, np.inf, 3])`
- Large arrays: 1M+ elements: probably in stress test instead of unit test for speed
- Broadcasting edge cases (probably caught by jaxtyping)

### Backend Compatibility
- Test with NumPy, JAX, and array-api-strict. Add option for CuPy but only if installed
- Verify Array API compliance
- Check numerical equivalence across backends

## Troubleshooting Test Failures

1. Use **python-test** command for TDD workflow
2. Check test isolation (fixtures, parametrize)
3. Verify numerical tolerances are appropriate
4. Fix implementation, not tests (unless tests are wrong)

## Agent Support

- **python-reviewer** - Use for code review before committing
- **python-build-resolver** - Use for type/lint errors
