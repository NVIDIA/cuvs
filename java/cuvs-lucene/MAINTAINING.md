# Maintaining cuvs-lucene across Lucene releases

`cuvs-lucene` is published as one artifact per supported Lucene minor release (`cuvs-lucene-10.2` …
`cuvs-lucene-10.5`), all built from the same sources. This page lists exactly what to change when
Lucene publishes a new patch release (for example 10.5.2) or a new minor release (for example 10.6.0).
For how the layout works, see [Supporting several Lucene releases](README.md#supporting-several-lucene-releases).

All commands run from `java/cuvs-lucene/` unless stated otherwise, with `libcuvs` built and `cuvs-java`
installed in the local Maven repository (`./build.sh libcuvs java` from the repository root).

## A new patch release (example: 10.5.2)

Every artifact accepts any patch release of its minor (the version check in `LuceneCompat` compares only
`major.minor`), so users can already run `cuvs-lucene-10.5` on Lucene 10.5.2. Moving the build to the
new patch release makes the artifact depend on, and be tested against, it.

1. In `lucene-10.5/pom.xml`, set `<lucene.version>10.5.2</lucene.version>`.
2. If the patch release is for the newest minor, also update the places that pin it: the default
   `lucene.version` in the parent `pom.xml`, and `lucene.version` in `examples/java/cuvs-lucene/pom.xml`.
3. Run the tests of that module: `mvn -pl lucene-10.5 verify` (add `-am` if the parent is not installed).
4. If something no longer compiles or a test fails, Lucene broke its API or behaviour in a patch
   release, which is rare. Fix it in that module's `LuceneCompat` (`lucene-10.5/src/main/java`), or
   in a new `compat/` variant used only by that module (see step 2 of the minor release below), never
   in the shared `src/`. A module targets one patch release at a time: if the fix cannot work with
   both 10.5.1 and 10.5.2, raise the minimum in the documentation.

Nothing else needs to change: no back-compat indexes, no codecs, no CI, no docs besides the examples.

## A new minor release (example: 10.6.0)

Work through the steps in order. Steps 1–3 always apply; the others depend on what the release changed.

### 1. Create the module

1. Copy the newest module: `cp -r lucene-10.5 lucene-10.6` (then `rm -rf lucene-10.6/target`).
2. In `lucene-10.6/pom.xml`:
   - change `artifactId` and `name` to `cuvs-lucene-10.6`, and the version in `description`;
   - set `<lucene.version>10.6.0</lucene.version>`;
   - keep the `add-since-sources` list: it must name the `src/since/java` directory of **every** earlier
     module that has one (today `../lucene-10.3/src/since/java` and `../lucene-10.4/src/since/java`);
   - keep the `cuvs.lucene.compat.*` properties: the new release starts with the same variant of each
     Lucene API as the previous one (see [compat/README.md](compat/README.md)).
3. In the parent `pom.xml`, add `<module>lucene-10.6</module>`, and set its default `lucene.version`
   to `10.6.0` (only the parent uses it; it tracks the newest supported release).
4. In `lucene-10.6/src/main/java/com/nvidia/cuvs/lucene/LuceneCompat.java`, set `LUCENE_MINOR = 6`
   and the Lucene version in the javadoc. Keep `LUCENE_MAJOR`/`LUCENE_MINOR` compile-time constants:
   the version check reads them without loading `LuceneCompat`.

`ci/release/update-version.sh`, `build.sh`, the CI workflows (`java/cuvs-lucene/*/target/`) and
`ci/test_lucene_prebuilt.sh` all match `lucene-*`. The one exception is `ci/test_lucene_jdk21.sh`,
which runs only the newest module on JDK 21: point its `-pl lucene-10.latest` at the new module.

### 2. Make it compile

```bash
mvn -pl lucene-10.6 -am test-compile
```

Fix every error without changing what the other modules compile:

- calls whose signature or class location changed (a class moved to `org.apache.lucene.backward_codecs`,
  a constructor lost or gained parameters, an enum moved to another class): in
  `lucene-10.6/src/main/java/.../LuceneCompat.java`;
- a Lucene class cuvs-lucene adapts in `compat/` changed (for example `KnnVectorsWriter`): copy the
  variant the module uses to `compat/<class>-106-<change>` (for example
  `compat/knn-vectors-writer-106-<change>`), change it there, and point the module's property
  (`cuvs.lucene.compat.knnVectorsWriter`) at it. Do not edit a variant other modules still use;
- a Lucene class with no variants yet changed: move its adapter from the shared `src/` into two new
  variants, `<class>-102-<old shape>` for the existing modules and `<class>-106-<change>`, add a
  `cuvs.lucene.compat.<class>` property to every module and a matching `<source>` line to the
  `add-shared-sources` execution of the parent `pom.xml` (see [compat/README.md](compat/README.md));
- test-only calls, such as `LeafReader.searchNearestVectors`: the same, with `TestLuceneCompat` in the
  `leaf-reader-*` variants (their `src/test/java`).

The shared `src/` must keep compiling against every supported release. If a shared class needs an API
that differs between releases, add a method to **every** module's `LuceneCompat` and call that instead.
Never use reflection, and never read version constants such as `VERSION_CURRENT` from Lucene: they
change meaning between releases.

### 3. Look for changes the compiler cannot see

A method Lucene adds with a default implementation compiles fine but may be wrong for cuvs-lucene (10.3's
`KnnVectorsReader.getOffHeapByteSize` is an example). Compare the overridable API of every Lucene class
cuvs-lucene extends between the previous and the new release:

```bash
M=~/.m2/repository/org/apache/lucene/lucene-core
for cls in org.apache.lucene.codecs.FilterCodec org.apache.lucene.codecs.KnnVectorsFormat \
    org.apache.lucene.codecs.KnnVectorsReader org.apache.lucene.codecs.KnnVectorsWriter \
    org.apache.lucene.codecs.KnnFieldVectorsWriter org.apache.lucene.search.KnnFloatVectorQuery \
    org.apache.lucene.search.TopKnnCollector org.apache.lucene.search.Query \
    org.apache.lucene.search.Weight org.apache.lucene.search.Scorer \
    org.apache.lucene.search.ScorerSupplier org.apache.lucene.search.DocIdSetIterator \
    'org.apache.lucene.util.hnsw.HnswGraph' 'org.apache.lucene.util.hnsw.HnswGraph$NodesIterator'; do
  diff <(javap -protected -cp $M/10.5.1/lucene-core-10.5.1.jar "$cls" | grep -Ev ' (final|static) ' | sort) \
       <(javap -protected -cp $M/10.6.0/lucene-core-10.6.0.jar "$cls" | grep -Ev ' (final|static) ' | sort) \
    | grep '^[<>]' && echo "^^^ $cls"
done
```

For each new or changed method, decide whether Lucene's default is right for the cuvs-lucene subclass.
If not, override it in the `Compat*` class of the variant the new release uses, creating a new variant
if older releases share that one (see how `compat/knn-vectors-reader-103-acceptdocs` overrides `getOffHeapByteSize`). To refresh the class list, run `javap` on the classes in
`lucene-10.6/target/classes/com/nvidia/cuvs/lucene` and collect their `org.apache.lucene` supertypes.

Also read Lucene's `CHANGES.txt` for the release, looking for:

- changes to `Lucene99HnswVectorsFormat`'s on-disk format. The GPU writers write its `.vex`/`.vem` files
  themselves (`AcceleratedHNSWUtils`) with `VERSION_START`; the new release's reader must still accept
  that version. If Lucene replaced `Lucene99HnswVectorsFormat` as its HNSW format, plan a new
  cuvs-lucene format; that is feature work, not part of this checklist;
- vector formats moved to `backward_codecs` (they can then only be read, see step 4);
- changes to `BaseKnnVectorsFormatTestCase` (see step 5).

### 4. If Lucene's default codec changed

Check which codec the release writes by default:

```bash
unzip -p ~/.m2/repository/org/apache/lucene/lucene-core/10.6.0/lucene-core-10.6.0.jar \
  META-INF/services/org.apache.lucene.codecs.Codec
```

If it is still `Lucene104Codec`, skip this step: the `Lucene104*` codecs keep writing on 10.6.
If it is new, for example `Lucene106Codec`:

1. Create `lucene-10.6/src/since/java/com/nvidia/cuvs/lucene/` with the four codecs of the new
   generation, copied from `lucene-10.4/src/since/java` and renamed: `Lucene106AcceleratedHNSWCodec`,
   `Lucene106AcceleratedHNSWScalarQuantizedCodec`, `Lucene106AcceleratedHNSWBinaryQuantizedCodec`,
   `Lucene106CuVSGPUSearchCodec`. Each passes generation `106` and `() -> LuceneCompat.lucene106Codec()`
   (a lambda, not a method reference) to the `CuVSFilterCodec` constructor. The codec name
   is written into every segment, so never reuse or rename one once released.
2. Add `src/since/java` to the `add-since-sources` list in `lucene-10.6/pom.xml` (and, later, in every
   module after it).
3. In `lucene-10.6/.../LuceneCompat.java`:
   - add `lucene106Codec()` returning `new org.apache.lucene.codecs.lucene106.Lucene106Codec()`;
   - point `lucene104Codec()` at `org.apache.lucene.backward_codecs.lucene104.Lucene104Codec`, where
     Lucene moves the old default codec;
   - set `CURRENT_CODEC_GENERATION = 106`. The new codecs pass `106` as their generation to the
     `CuVSFilterCodec` constructor, and the `Lucene104*` codecs, which pass `104`, become read-only;
   - make `acceleratedHNSWCodec`, `acceleratedHNSWScalarQuantizedCodec`,
     `acceleratedHNSWBinaryQuantizedCodec` and `gpuSearchCodec` return the `Lucene106*` codecs.
4. Register the four new codecs in `lucene-10.6/src/main/resources/META-INF/services/org.apache.lucene.codecs.Codec`,
   after the existing ones. List only cuvs-lucene classes in the services files: a Lucene class there
   breaks every codec or format lookup in the JVM once a later release moves it.
5. Update the list of codecs per Lucene release in the class javadoc of `CuVSCodecs`.

If a vector format cuvs-lucene writes moved to `backward_codecs` (as `Lucene99ScalarQuantizedVectorsFormat`
did in 10.4), the codecs using it can no longer write on 10.6: add a new format next to the codecs of
step 1 (see `lucene-10.4/src/since/java/.../Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java`),
register it in `META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat`, and have the new codecs
and `LuceneCompat.acceleratedHNSWScalarQuantizedFormat` use it.

### 5. Make the tests pass

```bash
mvn -pl lucene-10.6 -am verify
```

- New abstract methods or new tests in Lucene's `BaseKnnVectorsFormatTestCase` are handled once, in
  `src/test/java/.../BaseCuVSKnnVectorsFormatTestCase.java`, which every format test extends, and without
  `@Override`, so that they still compile against older releases. See `supportsFloatVectorFallback()`
  (abstract since Lucene 10.4) and `testWriterByteVectorRamEstimate()` (added in 10.5). Hooks that only
  some formats implement go in their test: `TestQuantizedVectorsFormats` implements
  `getQuantizationBits()` and `simulateEmptyRawVectors()` for the scalar-quantized format of 10.4+.
- A test call whose signature changed goes through `TestLuceneCompat`, in a `compat/` variant (step 2).
- `TestBackCompat` fails if a codec from `CuVSCodecs` is not registered, or if an older codec can still
  write.

### 6. Add the back-compat indexes

```bash
mvn -pl lucene-10.6 test -Dtest=TestBackCompatIndices -Dtests.method=testCreateIndices \
  -Dtests.cuvs.createBackCompatIndices=$PWD/src/test/resources/backcompat
```

This writes `10.6-<codec>.zip` for each codec from `CuVSCodecs` except the GPU search codec. Commit them.
Every module from 10.6 on reads them, and 10.6 reads every earlier one, so step 7 checks that indexes
written with each earlier release still open.

### 7. Run everything

```bash
./build.sh --run-java-tests --build-java-examples
```

This builds and tests every module, installs them, and builds the examples.

### 8. Update the documentation and examples

| File | Change |
| --- | --- |
| `README.md` (this directory) | the supported range and the artifact list; the codec paragraph if step 4 applied |
| `../../fern/pages/installation/java.md` | "supports Lucene 10.2 through 10.X", the artifact table, the dependency example |
| `../../fern/pages/user_guide/lucene.md` | the supported range, the artifact list, the codec table if step 4 applied |
| `../../examples/java/cuvs-lucene/pom.xml` | `lucene.version` and `cuvs.lucene.artifactId`, to use the newest release |
| `../../examples/java/cuvs-lucene/README.md` | the Lucene version the examples use |
| `bench/pom.xml` | the `cuvs-lucene-10.X` dependency, to use the newest release |

The API reference pages in `fern/pages/lucene_api` are generated: the `fern-api-reference` pre-commit
hook picks up new classes in `lucene-*/src/since/java` by itself. Run `pre-commit run --all-files` last.

## After a cuvs-lucene release

Add the indexes written by the published jars to the back-compat tests, so that later changes cannot
silently make indexes written by that release unreadable. On a host with a GPU and the matching native
libraries:

```bash
src/test/backcompat/generate-released-indices.sh <released version>
```

This currently handles releases published as the single `cuvs-lucene` artifact (up to 26.08); releases
from 26.12 on are published per Lucene release, so the script needs extending to pick one of those
artifacts. Record the new zips in `src/test/resources/backcompat/README.md`.

## Dropping a minor release

1. Remove its `<module>` from the parent `pom.xml` and delete its `lucene-10.X` directory. If it has a
   `src/since/java`, move that directory to the oldest remaining module that lists it, and update the
   `add-since-sources` lists of the later modules.
2. Delete every `compat/` variant no remaining module uses (`grep -h cuvs.lucene.compat lucene-*/pom.xml`).
   If all remaining modules use the same variant of an API, move its classes into the shared `src/`,
   delete the variant directory and the property, and remove its `<source>` line from the parent
   `pom.xml`. Do the same for any `LuceneCompat` method that is now identical in every module.
3. Keep its codecs and its `src/test/resources/backcompat/10.X-*.zip` files as long as later releases
   must read indexes written with it.
4. Update the documentation listed in step 8 above.
5. When dropping 10.2, the relocation in `relocation/pom.xml` would point to an artifact that is no longer
   published: remove the `relocation` module (and stop publishing it) at that point. The deprecated
   `LuceneProvider` in `lucene-10.2/src/main/java`, kept only for callers of the old artifact, goes
   with the module.
