# Back-compat indexes

Indexes read and upgraded by `TestBackCompatIndices`, one zip each, named
`<lucene major.minor>-<codec name>[-<per-field format name>][-cuvs-<cuvs-lucene version>].zip`. They all
hold the same 200 documents (see `TestBackCompatIndices#writeIndex`). Every module reads the zips
written on its own Lucene release and on all earlier ones.

## Written by this repository

`<lucene>-<codec>.zip`: written by the `lucene-X.Y` module with each codec from `CuVSCodecs` except the
GPU search one. To add them for a new Lucene release, see `MAINTAINING.md` step 6:

```bash
mvn -pl lucene-10.X test -Dtest=TestBackCompatIndices -Dtests.method=testCreateIndices \
  -Dtests.cuvs.createBackCompatIndices=$PWD/src/test/resources/backcompat
```

## Written by released cuvs-lucene versions

`<lucene>-...-cuvs-<version>.zip`: written by the jars of a cuvs-lucene release published to Maven
Central, with `src/test/backcompat/generate-released-indices.sh <version>`. They catch changes that make
indexes written by released code unreadable. Releases up to 26.08 were a single
`com.nvidia.cuvs.lucene:cuvs-lucene` artifact, built for Lucene 10.2.

| Release | Written with | Notes |
| --- | --- | --- |
| 26.08.0 | each of its codecs, except the GPU search codec | built on the GPU |
| 25.12.0 | Lucene's default codec, with `Lucene99AcceleratedHNSWVectorsFormat` per field (the release registered no codecs; this is how Solr uses it) | built by the release's CPU fallback: its cuvs-java cannot load against libcuvs 26.06 or later, whose CUDA runtime is linked statically |
