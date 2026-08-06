---
slug: api-reference/rust-api-cuvs-test-utils
---

# Test Utils Module

_Rust module: `cuvs::test_utils`_

_Source: `rust/cuvs/src/test_utils.rs`_

Test-only tensor adapters.

[`DeviceTensor`] is an RMM-backed device matrix, and the `ndarray` host
adapters below implement [`AsDlTensor`]/[`AsDlTensorMut`] for plain host
arrays. We use `ndarray` only as a dev-dependency to assist with unit tests.
