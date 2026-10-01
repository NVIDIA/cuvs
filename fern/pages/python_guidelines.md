---
slug: developer-guide/coding-guidelines/python-guidelines
---

# Python Guidelines

This page collects Python-facing conventions for NVIDIA cuVS development. Start with the [Contributor Guide](/developer-guide/contributing), then use this page when changing Python APIs, Cython bindings, Python packaging, or user-facing Python documentation.

## Local Development

Most Python changes can be developed directly in this repository. Cross-project work may also require a local RAFT build or temporary downstream pin.

If source builds are not being used, install the local RAFT Python artifacts into the consuming project's environment before testing the downstream change.

## Public Interface

### Bindings

Python APIs, like all other NVIDIA cuVS language bindings, should wrap the C APIs and should not call C++ or CUDA implementation code directly. The C layer is the ABI-stable boundary for bindings, so ABI compatibility needs to be maintained there. See [ABI Stability](/developer-guide/advanced-topics/abi-stability) for more detail.

Keep Cython bindings focused on translating Python inputs into NVIDIA cuVS calls and returning Python-friendly outputs. Heavy algorithmic work should stay in C++ or CUDA implementation code.

## Coding Style

### Formatting

NVIDIA cuVS uses [pre-commit](https://pre-commit.com/) to run formatting, linting, spelling, and copyright checks. Install it with conda or pip:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Run checks before committing:

```bash
pre-commit run
```

Run the full suite across the repository when needed:

```bash
pre-commit run --all-files
```

You can also install the git hook:

```bash
pre-commit install
```

Python code is checked by repository hooks such as [isort](https://pycqa.github.io/isort/), Ruff, mypy, pydocstyle, and spelling checks.

## Code Quality

### Testing

Python APIs need direct test coverage because downstream projects rely on their runtime behavior. Prefer tests that exercise public entry points, common input types, error paths, and expected output shapes without depending on private implementation details.

### Documentation

Python and Cython APIs use [pydoc](https://docs.python.org/3/library/pydoc.html). Document the purpose, parameters, return values, and constraints that affect correct use.
