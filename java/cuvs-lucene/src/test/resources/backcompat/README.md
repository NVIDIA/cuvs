# Back-compat indexes

Indexes read and upgraded by `TestBackCompatIndices`, one zip each. They all hold the same 200 documents
(see `TestBackCompatIndices#writeIndex`). Every module reads the zips written on its own Lucene release and
on all earlier ones. The directories record who wrote each index and on which Lucene release:

```
releases/<cuvs-lucene version>/<lucene major.minor>/<codec>.zip   written by a published release
releases/<cuvs-lucene version>/<lucene major.minor>/<codec>-<per-field format>.zip
snapshots/<lucene major.minor>/<codec>.zip                         written by this repository
```

`<codec>` is the codec recorded in the index's segments. When that is Lucene's own codec, `<per-field format>`
is the cuvs-lucene vectors format its vector field uses. `TestBackCompatIndices` fails on a zip placed
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

## Snapshots: Lucene releases that no published version supports yet

`snapshots/<lucene>/`: written by this repository's `lucene-X.Y` module, with each codec from `CuVSCodecs`
except the GPU search one. They exist only so that a newly added Lucene release is tested by the later
modules before any cuvs-lucene release has written indexes on it. Add them in the change that adds the
module:

```bash
mvn -pl lucene-10.X test -Dtest=TestBackCompatIndices -Dtests.method=testCreateIndices \
  -Dtests.cuvs.createBackCompatIndices=$PWD/src/test/resources/backcompat
```

Once a published release supports that Lucene release and its `releases/<version>/<lucene>/` fixtures are
added, delete `snapshots/<lucene>/`. Don't regenerate snapshots otherwise: a fixture that is rewritten with
each change can no longer catch a change that makes older indexes unreadable.
