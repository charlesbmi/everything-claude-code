---
name: poe-task-runner
description: Guide for using Poe the Poet task runner with uv for Python project automation. Covers linting, testing, and custom tasks.
---

# Poe the Poet Task Runner

Task automation for Python projects using [Poe the Poet](https://poethepoet.natn.io/) with `uv` for dependency management.

## When to Activate

- Project has `[tool.poe.tasks]` in `pyproject.toml`
- Project has `poe_tasks.toml`, `poe_tasks.yaml`, or `poe_tasks.json` in root
- User mentions "poe", "poethepoet", or asks about task runners
- Setting up project automation with uv

## Quick Reference

```bash
# Install dependencies (includes poe)
uv sync

# List available tasks
uv run poe --help

# Run a task
uv run poe <task-name>

# Common task patterns
uv run poe lint      # Lint and format
uv run poe test      # Run tests
uv run poe test-all  # Run all tests with coverage
uv run poe benchmark # Run benchmarks
```

## Configuration in pyproject.toml

### Basic Task Definition

```toml
[tool.poe.tasks]
# Simple command
lint = "ruff check --fix src tests"

# Sequence of commands
format = ["ruff format src tests", "ruff check --fix src tests"]

# Command with description
test = { cmd = "pytest", help = "Run test suite" }
```

### Task Sequences

```toml
[tool.poe.tasks]
# Run multiple commands in sequence
lint = [
    { cmd = "ruff format src tests" },
    { cmd = "ruff check --fix src tests" },
    { cmd = "ty check" },
]

# Or using ref to compose tasks
_format = "ruff format src tests"
_check = "ruff check --fix src tests"
_typecheck = "ty check"
lint = { sequence = ["_format", "_check", "_typecheck"] }
```

### Common Task Patterns

```toml
[tool.poe.tasks]
# Linting sequence
lint = [
    { cmd = "ruff format src tests" },
    { cmd = "ruff check --fix src tests" },
    { cmd = "ty check" },
    { cmd = "codespell src tests docs README.md" },
]

# Fast tests (affected only with testmon)
test = { cmd = "pytest -n auto --testmon", help = "Run affected tests only" }

# Full test suite with coverage
test-all = { cmd = "pytest -n auto --cov=src --cov-report=xml", help = "Run all tests with coverage" }

# Benchmarks
benchmark = { cmd = "pytest --benchmark-only --benchmark-autosave", help = "Run benchmarks" }

# Documentation
docs = { cmd = "mkdocs serve", help = "Serve documentation locally" }
docs-build = { cmd = "mkdocs build", help = "Build documentation" }
```

### Tasks with Arguments

```toml
[tool.poe.tasks]
# Pass arguments to underlying command
test = { cmd = "pytest", help = "Run tests (args passed to pytest)" }
# Usage: uv run poe test -k "test_name" -v

# Explicit argument handling
[tool.poe.tasks.run]
cmd = "python -m mymodule"
args = [{ name = "config", options = ["-c", "--config"], help = "Config file" }]
```

### Environment Variables

```toml
[tool.poe.tasks]
test = { cmd = "pytest", env = { PYTHONDONTWRITEBYTECODE = "1" } }

# Or load from .env file
[tool.poe]
envfile = ".env"
```

## Integration with Common Tools

### Ruff (Linting/Formatting)

```toml
[tool.poe.tasks]
format = "ruff format src tests"
check = "ruff check --fix src tests"
lint = { sequence = ["format", "check"] }

[tool.ruff]
target-version = "py311"
line-length = 120
fix = true

[tool.ruff.lint]
select = [
    "B",     # flake8-bugbear
    "E",     # pycodestyle errors
    "F",     # pyflakes
    "I",     # isort
    "RUF",   # ruff-specific
    "S",     # flake8-bandit (security)
    "SIM",   # flake8-simplify
    "UP",    # pyupgrade
    "W",     # pycodestyle warnings
]
```

### Type Checking (ty or mypy)

```toml
[tool.poe.tasks]
typecheck = "ty check"           # Fast Rust-based checker
# Or: typecheck = "mypy src"     # Traditional mypy
```

### pytest with testmon

```toml
[tool.poe.tasks]
# Fast: only run affected tests
test = "pytest -n auto --testmon"

# Full suite
test-all = "pytest -n auto --cov=src --cov-report=xml"

# Run specific marker
test-slow = "pytest -m slow"
test-fast = "pytest -m 'not slow'"

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: requires GPU",
]
```

### pytest-benchmark

```toml
[tool.poe.tasks]
benchmark = "pytest --benchmark-only --benchmark-autosave"
benchmark-compare = "pytest --benchmark-only --benchmark-compare"
```

## Pre-commit Integration

```toml
[tool.poe.tasks]
# Install pre-commit hooks
hooks-install = "pre-commit install"

# Run hooks on all files
hooks-run = "pre-commit run --all-files"
```

## Troubleshooting

### Task Not Found

```bash
# List all available tasks
uv run poe --help
```

### testmon Cache Issues

```bash
# Clear testmon cache
rm .testmondata*
uv run poe test
```

### Dependency Issues

```bash
# Reinstall dependencies
uv sync --reinstall
```

## Example Full Configuration

```toml
[tool.poe.tasks]
# Linting
format = "ruff format src tests"
check = "ruff check --fix src tests"
typecheck = "ty check"
spell = "codespell src tests docs README.md"
lint = { sequence = ["format", "check", "typecheck", "spell"], help = "Run all linters" }

# Testing
test = { cmd = "pytest -n auto --testmon", help = "Run affected tests only" }
test-all = { cmd = "pytest -n auto --cov=src --cov-report=xml", help = "Full test suite with coverage" }
benchmark = { cmd = "pytest --benchmark-only --benchmark-autosave", help = "Run benchmarks" }

# Documentation
docs = { cmd = "mkdocs serve", help = "Serve docs locally" }
docs-build = { cmd = "mkdocs build", help = "Build documentation" }

# Pre-commit
hooks = "pre-commit run --all-files"
```

## Best Practices

1. **Prefix internal tasks with underscore** - `_format` won't show in `--help`
2. **Add help descriptions** - Makes `poe --help` useful
3. **Use sequences for multi-step tasks** - Ensures consistent execution order
4. **Keep task names short** - `lint`, `test`, `docs` not `run-all-linters`
5. **Match CI commands** - Same tasks run locally and in CI
