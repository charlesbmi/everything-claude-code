---
description: Fix Python type errors, linting issues, and import problems incrementally. Invokes the python-build-resolver agent for minimal, surgical fixes.
---

# Python Build and Fix

This command invokes the **python-build-resolver** agent to incrementally fix Python type errors and linting issues with minimal changes.

## What This Command Does

1. **Run Diagnostics**: Execute `ty check`, `ruff check`, `pytest --collect-only`
2. **Parse Errors**: Group by file and sort by severity
3. **Fix Incrementally**: One error at a time
4. **Verify Each Fix**: Re-run checks after each change
5. **Report Summary**: Show what was fixed and what remains

## When to Use

Use `/python-build` when:
- `ty check src/` fails with type errors
- `ruff check .` reports linting issues
- Import errors prevent code from running
- Dependency conflicts in `uv.lock`
- After pulling changes that break type checking

## Diagnostic Commands Run

```bash
# Type checking
ty check src/

# Linting
ruff check .

# Import validation
python -c "import my_library"

# Dependency check
uv sync
uv lock --check

# Test collection
pytest --collect-only
```

## Example Session

```text
User: /python-build

Agent:
# Python Build Resolution

## Initial Diagnostics

```bash
$ ty check src/

src/fast_simus/pfield.py:42: error: Missing type annotation
src/fast_simus/validation.py:15: error: Name 'NDArray' is not defined
src/fast_simus/transforms.py:28: error: Incompatible return type
```

Errors found: 3

## Fix 1: Missing Type Annotation

File: src/fast_simus/pfield.py:42
Error: Missing type annotation for function parameter
Cause: No type hint on function signature

```python
# Before
def compute_pfield(positions, frequency):
    return positions * frequency

# After
from jaxtyping import Float
from numpy.typing import NDArray

def compute_pfield(
    positions: Float[NDArray, "points 3"],
    frequency: float
) -> Float[NDArray, "points"]:
    return np.linalg.norm(positions, axis=1) * frequency
```

```bash
$ ty check src/
src/fast_simus/validation.py:15: error: Name 'NDArray' is not defined
src/fast_simus/transforms.py:28: error: Incompatible return type
```

Remaining errors: 2

## Fix 2: Missing Import

File: src/fast_simus/validation.py:15
Error: Name 'NDArray' is not defined
Cause: Missing import from numpy.typing

```python
# Added import
from numpy.typing import NDArray

# Now type hint works
def validate_signal(signal: NDArray) -> None:
    pass
```

```bash
$ ty check src/
src/fast_simus/transforms.py:28: error: Incompatible return type
```

Remaining errors: 1

## Fix 3: Incompatible Return Type

File: src/fast_simus/transforms.py:28
Error: Expected 'NDArray', got 'float'
Cause: Function returns scalar instead of array

```python
# Before
def compute_norm(x: NDArray) -> NDArray:
    return np.sqrt(np.sum(x ** 2))  # Returns scalar!

# After
def compute_norm(x: NDArray) -> float:
    return float(np.sqrt(np.sum(x ** 2)))
```

```bash
$ ty check src/
Success: no issues found
```

## Build Status: SUCCESS

Errors Fixed: 3
Files Modified: 3
- src/fast_simus/pfield.py
- src/fast_simus/validation.py
- src/fast_simus/transforms.py

## Verification

```bash
$ ruff check .
All checks passed!

$ pytest --collect-only
collected 42 items
```

All systems operational!
```

## Common Error Patterns

### Type Errors

```python
# Error: Missing type hint
def process(signal):  # ❌
    pass

def process(signal: NDArray) -> NDArray:  # ✅
    pass

# Error: Wrong return type
def get_value() -> int:
    return "string"  # ❌

def get_value() -> str:
    return "string"  # ✅
```

### Import Errors

```python
# Error: Module not found
from nonexistent import func  # ❌

# Fix: Install dependency
# $ uv add package-name

# Error: Circular import
# a.py imports b.py
# b.py imports a.py  # ❌

# Fix: Move shared code to separate module
# shared.py contains common code
# a.py and b.py both import shared.py  # ✅
```

### Linting Issues

```python
# Error: Unused import
import numpy as np  # ❌ (if not used)

# Fix: Remove or use it
import numpy as np
result = np.array([1, 2, 3])  # ✅

# Error: Line too long
very_long_function_call(argument1, argument2, argument3, argument4, argument5)  # ❌

# Fix: Break into multiple lines
very_long_function_call(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5
)  # ✅
```

## Auto-Fix Capabilities

```bash
# Auto-fix safe linting issues
ruff check --fix .

# Auto-fix specific rules
ruff check --select F401 --fix .  # Remove unused imports

# Format code
ruff format .
```

## Dependency Issues

```bash
# Sync dependencies
uv sync

# Update lock file
uv lock --upgrade

# Add missing dependency
uv add numpy

# Remove dependency
uv remove old-package

# Check for conflicts
uv lock --check
```

## Stop Conditions

The agent will stop and report if:
- Same error persists after 3 fix attempts
- Fix introduces more errors than it resolves
- Error requires architectural changes
- Circular dependency needs restructuring
- Missing external dependency needs manual installation

## Resolution Strategy

1. **Read error message** - Python errors are descriptive
2. **Identify file and line** - Go directly to source
3. **Understand context** - Read surrounding code
4. **Make minimal fix** - Don't refactor, just fix
5. **Verify fix** - Run checks again
6. **Repeat** - Continue until clean

## Related Commands

- `/python-test` - Run tests after build succeeds
- `/python-review` - Review code quality
- `/build-fix` - General build fixing (language-agnostic)

## Related

- Agent: `agents/python-build-resolver.md`
- Skills: `skills/python-scientific-patterns/`
