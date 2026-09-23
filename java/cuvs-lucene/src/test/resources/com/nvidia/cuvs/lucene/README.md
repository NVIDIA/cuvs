# Released binary-vector index fixture

`cuvs-lucene-26.08.0-binary-index.zip.b64` is a Base64-encoded Lucene index
created on an NVIDIA A10G by the published
`com.nvidia.cuvs.lucene:cuvs-lucene:26.08.0` artifact. The artifact SHA-256 is
`12fab622a94431a81d9b99f9816370c6009f5dc736a0c5d3de811f82ae570f16`.
It used `com.nvidia.cuvs:cuvs-java:26.08.0` (SHA-256
`820a244b916b83003262784fea47b9faab01b2c7f80a0abc30b48791ec09b6a1`)
and the cached `libcuvs-26.08.00a82-cuda12_260707_599fbb43` runtime.

The non-compound index contains 128 documents. Document `i` stores integer
field `id=i` and a 128-dimensional Euclidean vector whose elements `i % 128`
and `(i * 7 + 3) % 128` are `1.0` and `0.5`, respectively. It was written with
graph degree 32, intermediate graph degree 64, one HNSW layer, and the released
binary accelerated codec. The
unencoded archive SHA-256 is
`157c4711b083d1985f78424e1543008abe8111fdb887d83369be983d6bd06f69`.

The fixture intentionally has Lucene99 `.vemf`/`.vec` flat-vector files and no
NVIDIA layout attribute. It protects the new-reader/old-index contract without
requiring an old writer or GPU during the compatibility test.
