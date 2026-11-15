# Attribution

This task is inspired by the hnswlib filtered-search example and preserves its
even-id filtered-neighbor contract with original local fixture code.

- Source: https://github.com/nmslib/hnswlib/blob/d9b3608c83d8/examples/python/example_filter.py
- Upstream project: hnswlib
- License: Apache-2.0
- Local changes: replaced the hnswlib runtime dependency with a NumPy baseline
  so benchmark setup does not require compiling hnswlib.
