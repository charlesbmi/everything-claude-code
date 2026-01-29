# Build and Fix

Incrementally fix Python type errors and linting issues:

1. Run checks: `ty check src/` and `ruff check .`

2. Parse error output:
   - Group by file
   - Sort by severity

3. For each error:
   - Show error context (5 lines before/after)
   - Explain the issue
   - Propose fix
   - Apply fix
   - Re-run checks
   - Verify error resolved

4. Stop if:
   - Fix introduces new errors
   - Same error persists after 3 attempts
   - User requests pause

5. Show summary:
   - Errors fixed
   - Errors remaining
   - New errors introduced

Fix one error at a time for safety!

## Common Python Error Types

- Type errors (ty check)
- Missing imports
- Linting issues (ruff)
- Array shape mismatches
- Dependency conflicts (uv)
