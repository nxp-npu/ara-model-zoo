# Test Execution Guide

This document describes the recommended pytest commands for executing the project's test suites and generating code coverage reports.

## Prerequisites

- Python environment activated
- Project dependencies installed
- Execute all commands from the project root directory

---

# 1. Run Test Suites (Without Coverage)

Use these commands to execute the test suites without collecting code coverage. They are suitable for local development, pre-commit verification, and CI pipelines where only test execution and validation are required.

> **Note**
>
> - `-v` enables verbose test output.
> - `-vv` enables extra verbose output, providing more detailed information during test discovery and execution. It is particularly useful for debugging and can also be used in CI pipelines when more detailed logs are desired.

## Run the Complete Test Suite

Execute all available tests.

```bash
pytest tests -v
```

## Run Individual Test Suites

Execute only the tests associated with a specific component.

### Preprocessing

```bash
pytest tests/test_preprocessing -vv
```

### Postprocessing

```bash
pytest tests/test_postprocessing -vv
```

### Configuration Schema

```bash
pytest tests/test_config_schema -vv
```

---

# 2. Run Individual Modules with Coverage

Generate coverage reports for a specific module.

## Preprocessing

```bash
pytest tests/test_preprocessing --cov=core/python/preprocess --cov-report=term-missing -v
```

## Postprocessing

```bash
pytest tests/test_postprocessing --cov=core/python/postprocess --cov-report=term-missing -v
```

## Inference

```bash
pytest tests/test_inference --cov=core/python/inference --cov-report=term-missing -v
```

## Configuration Schema

```bash
pytest tests/test_config_schema --cov=core/python/config --cov-report=term-missing -v
```

---

# 3. Run Multiple Core Components with Coverage

Execute preprocessing, configuration schema, and postprocessing tests together.

```bash
pytest tests/test_preprocessing tests/test_config_schema tests/test_postprocessing --cov=core/python --cov-report=term-missing -v
```

---

# 4. Run Complete Test Suite with Coverage

Execute every available test and generate an overall coverage report.

```bash
pytest tests --cov=core/python --cov-report=term-missing -v
```

---

## Coverage Report

All coverage commands use:

- `--cov` — measures code coverage.
- `--cov-report=term-missing` — prints missing lines directly in the terminal.
- `-v` — verbose execution output.
- `-vv` — extra verbose output for debugging and development.
