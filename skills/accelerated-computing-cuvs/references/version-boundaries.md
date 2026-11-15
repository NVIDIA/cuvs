# cuVS Version Boundaries

Use this file when the installed cuVS version, local checkout, or public docs
do not match the skill's tracked release. Prefer the installed package's
docstrings and official docs for exact signatures.

## Tracked Release: 26.06

The main skill guidance targets cuVS 26.06. For Python vector-search APIs in
26.06:

| API | Python input dtype boundary |
|---|---|
| `cuvs.neighbors.cagra` | `float32`, `float16`, `int8`, `uint8` |
| `cuvs.neighbors.ivf_flat` | `float32`, `float16`, `int8`, `uint8` |
| `cuvs.neighbors.ivf_pq` | `float32`, `float16`, `int8`, `uint8`; search also has `lut_dtype`, `internal_distance_dtype`, and `coarse_search_dtype` knobs |
| `cuvs.neighbors.hnsw` | `float32`, `float16`, `int8`, `uint8`; `hierarchy="gpu"` is available |
| `cuvs.neighbors.brute_force` | `float32`, `float16` |
| `cuvs.neighbors.refine` | `float32`, `float16`, `int8`, `uint8` inputs with `float32` distances |

Use `float32` as the dense embedding baseline. Choose `float16`, `int8`, or
`uint8` when the source embeddings already use that representation, the memory
budget requires it, and recall validation confirms the target quality.

The current public Python API accepts `metric="cosine"` on CAGRA and IVF-Flat
`IndexParams`, along with `sqeuclidean` and `inner_product`. IVF-Flat also
accepts `euclidean`. Use `metric="cosine"` when the installed `IndexParams`
documents it. When that value is rejected, L2-normalize the dataset and the
queries and use `metric="inner_product"`. Do not assume IVF-PQ accepts
`cosine` unless the installed signature lists it.

Python 26.06 includes the higher-level modules used by the skill examples:
`cagra`, `ivf_flat`, `ivf_pq`, `hnsw`, `brute_force`, `filters`, `refine`, and
multi-GPU variants under `cuvs.neighbors.mg`.

## Stable 26.04 Boundary

The checked 26.04 Python dtype/API boundaries for the skill's main examples
match the 26.06 guidance above: CAGRA, IVF-Flat, IVF-PQ, HNSW, filters, refine,
and multi-GPU modules are present, and the documented input dtype sets for
those examples are the same. Keep 26.04-specific install/platform claims out of
the 26.06 top-level skill text.

## Old Release Boundary: 24.04

Keep 24.04-specific compatibility notes here instead of in the top-level skill.
The 24.04 Python package had a narrower vector-search surface:

- Python neighbor modules included `cagra` and `brute_force`; Python `ivf_flat`,
  `ivf_pq`, `hnsw`, `filters`, `refine`, and `mg` were not present in the same
  package layout.
- Python CAGRA accepted `float32`, `int8`, and `uint8` inputs.
- Python CAGRA did not accept `float16` in 24.04.
- Python brute-force enforced `float32` inputs.
- IVF-Flat and IVF-PQ existed in the 24.04 C and C++ documentation, so a
  24.04 user may need C/C++ APIs for those paths rather than Python wrappers.

For 24.04 migrations, choose examples that match the exposed Python modules or
route language-specific work through the C/C++ API docs for that release.

## Newer Docs Boundary

The public documentation at `https://docs.nvidia.com/cuvs/home` may describe newer
releases than this skill's tracked 26.06 baseline. Newer docs can include
additional index families or language bindings. Use those docs for current
product positioning and installed-version signatures, then state the installed
`cuvs.__version__` when version-sensitive APIs or dtypes matter.

The C shared library is the stable binary boundary for database/search-engine
integrations and language bindings. C++ API consumers should build against the
target cuVS/CUDA/compiler/dependency set instead of assuming C++ ABI stability.
