# Back-compat indexes

Indexes read and upgraded by `TestBackCompatIndices`, one zip each. They all hold the same 200 documents
(see `GenerateReleasedIndices#writeIndex` in `src/test/backcompat`). Every module reads the zips written on its own Lucene release and
on all earlier ones. The directories record which release wrote each index and on which Lucene release:

```
releases/<cuvs-lucene version>/<lucene major.minor>/<name>.zip
```

`<name>` is the cuvs-lucene codec recorded in the index's segments or, for an index written with Lucene's own
codec, the cuvs-lucene vectors format its vector field uses. `TestBackCompatIndices` fails on a zip placed
anywhere else, so that it can't go unread.

## Written by released cuvs-lucene versions

`releases/<version>/`: written by the jars of a cuvs-lucene release published to Maven Central, with
`src/test/backcompat/generate-released-indices.sh <version>`. They catch changes that make indexes written
by released code unreadable. Releases up to 26.10 were a single `com.nvidia.cuvs.lucene:cuvs-lucene`
artifact, built for Lucene 10.2.

| Release | Written with | Notes |
| --- | --- | --- |
| 26.10.0 | each of its codecs, except the GPU search codec | built on the GPU, with the 26.10 native libraries |
| 26.08.0 | each of its codecs, except the GPU search codec | built on the GPU |
| 25.12.0 | Lucene's default codec, with `Lucene99AcceleratedHNSWVectorsFormat` per field (the release registered no codecs; this is how Solr uses it) | built by the release's CPU fallback: its cuvs-java cannot load against libcuvs 26.06 or later, whose CUDA runtime is linked statically |
