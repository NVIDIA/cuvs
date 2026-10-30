# Maintaining cuvs-lucene across Lucene releases

`cuvs-lucene` is published as one artifact per supported Lucene minor release (`cuvs-lucene-10.2`,
`cuvs-lucene-10.3`), all built from the same sources. This page lists exactly what to change when Lucene
publishes a new patch release (for example 10.3.3) or a new minor release (for example 10.4.0). For how
the layout works, see [Supporting several Lucene releases](README.md#supporting-several-lucene-releases).

All commands run from `java/cuvs-lucene/` unless stated otherwise, with `libcuvs` built and `cuvs-java`
installed in the local Maven repository (`./build.sh libcuvs java` from the repository root).

## A new patch release (example: 10.3.3)

Every artifact accepts any patch release of its minor (the version check compares only `major.minor`),
so users can already run `cuvs-lucene-10.3` on Lucene 10.3.3. Moving the build to the new patch release
makes the artifact depend on, and be tested against, it.

1. In `lucene-10.3/pom.xml`, set `<lucene.version>10.3.3</lucene.version>`.
2. Run the tests of that module: `mvn -pl lucene-10.3 verify` (add `-am` if the parent is not installed).
3. If something no longer compiles or a test fails, Lucene broke its API or behaviour in a patch
   release, which is rare. Fix it in that module's `LuceneCompat` (`lucene-10.3/src/main/java`), or
   in a new `compat/` variant used only by that module (see step 2 of the minor release below), never
   in the shared `src/`. A module targets one patch release at a time: if the fix cannot work with
   both 10.3.2 and 10.3.3, raise the minimum in the documentation.

Nothing else needs to change: no back-compat indexes, no codecs, no CI, no docs.

## A new minor release (example: 10.4.0)

Work through the steps in order. Steps 1–3 always apply; the others depend on what the release changed.

### 1. Create the module

1. Copy the newest module: `cp -r lucene-10.3 lucene-10.4`, then `rm -rf lucene-10.4/target
   lucene-10.4/src/since`. The `src/since/java` of the copied module belongs to that module only;
   later modules compile it from there (below), and step 4 adds the new module's own if it needs one.
2. In `lucene-10.4/pom.xml`:
   - change `artifactId` and `name` to `cuvs-lucene-10.4`, and the version in `description`;
   - set `<lucene.version>10.4.0</lucene.version>`;
   - keep the `add-since-sources` list, but name the `src/since/java` directory of **every** earlier
     module that has one instead of its own (today `../lucene-10.3/src/since/java`); step 4 adds its
     own if it gets one;
   - keep the `cuvs.lucene.compat.*` properties: the new release starts with the same variant of each
     Lucene API as the previous one (see [compat/README.md](compat/README.md)).
3. In the parent `pom.xml`, add `<module>lucene-10.4</module>`.
4. In `lucene-10.4/src/main/java/com/nvidia/cuvs/lucene/LuceneCompat.java`, set `LUCENE_MINOR = 4`
   and the Lucene version in the javadoc. Keep `LUCENE_MAJOR`/`LUCENE_MINOR` compile-time constants:
   the version check reads them without loading `LuceneCompat`.

`ci/release/update-version.sh`, `build.sh` and the CI scripts and workflows (`java/cuvs-lucene/*/target/`)
all match `lucene-*`. `ci/test_lucene.sh --min-jdk` tests the newest `lucene-*` module, so it moves to
the new module by itself.

### 2. Make it compile

```bash
mvn -pl lucene-10.4 -am test-compile
```

Fix every error without changing what the other modules compile:

- calls whose signature or class location changed (a class moved to `org.apache.lucene.backward_codecs`,
  a constructor lost or gained parameters, an enum moved to another class): in
  `lucene-10.4/src/main/java/.../LuceneCompat.java`;
- a Lucene class cuvs-lucene adapts in `compat/` changed (for example `KnnVectorsReader`): copy the
  variant the module uses to `compat/<class>-104-<change>` (for example
  `compat/knn-vectors-reader-104-<change>`), change it there, and point the module's property
  (`cuvs.lucene.compat.knnVectorsReader`) at it. Do not edit a variant other modules still use;
- a Lucene class with no variants yet changed (for example `KnnVectorsWriter`): move its adapter from
  the shared `src/` into two new variants, `<class>-102-<old shape>` for the existing modules and
  `<class>-104-<change>`, add a `cuvs.lucene.compat.<class>` property to every module and a matching
  `<source>` line to the `add-shared-sources` execution of the parent `pom.xml` (see
  [compat/README.md](compat/README.md));
- test-only calls, such as `LeafReader.searchNearestVectors`: the same, with `TestLuceneCompat` in the
  `leaf-reader-*` variants (their `src/test/java`).

The shared `src/` must keep compiling against every supported release. If a shared class needs an API
that differs between releases, add a method to **every** module's `LuceneCompat` and call that instead.
Never use reflection, and never read version constants such as `VERSION_CURRENT` from Lucene: they
change meaning between releases.

### 3. Look for changes the compiler cannot see

A method Lucene adds with a default implementation compiles fine but may be wrong for cuvs-lucene (10.3's
`KnnVectorsReader.getOffHeapByteSize` is an example). Compare the overridable API of every Lucene class
cuvs-lucene extends, and of their superclasses, between the previous and the new release. `javap` lists
only the members a class declares itself, so the list names the superclasses too:

```bash
M=~/.m2/repository/org/apache/lucene/lucene-core
for cls in org.apache.lucene.codecs.FilterCodec org.apache.lucene.codecs.KnnVectorsFormat \
    org.apache.lucene.codecs.KnnVectorsReader org.apache.lucene.codecs.KnnVectorsWriter \
    org.apache.lucene.codecs.KnnFieldVectorsWriter org.apache.lucene.search.KnnFloatVectorQuery \
    org.apache.lucene.search.AbstractKnnVectorQuery org.apache.lucene.search.TopKnnCollector \
    org.apache.lucene.search.AbstractKnnCollector org.apache.lucene.search.KnnCollector \
    org.apache.lucene.search.Query \
    org.apache.lucene.search.Weight org.apache.lucene.search.Scorer \
    org.apache.lucene.search.ScorerSupplier org.apache.lucene.search.DocIdSetIterator \
    'org.apache.lucene.util.hnsw.HnswGraph' 'org.apache.lucene.util.hnsw.HnswGraph$NodesIterator'; do
  diff <(javap -protected -cp $M/10.3.2/lucene-core-10.3.2.jar "$cls" | grep -Ev ' (final|static) ' | sort) \
       <(javap -protected -cp $M/10.4.0/lucene-core-10.4.0.jar "$cls" | grep -Ev ' (final|static) ' | sort) \
    | grep '^[<>]' && echo "^^^ $cls"
done
```

For each new or changed method, decide whether Lucene's default is right for the cuvs-lucene subclass.
If not, override it in the `Compat*` class of the variant the new release uses, creating a new variant
if older releases share that one (see how `compat/knn-vectors-reader-103-acceptdocs` overrides
`getOffHeapByteSize`). To refresh the class list, run `javap` on the classes in
`lucene-10.4/target/classes/com/nvidia/cuvs/lucene` and collect their `org.apache.lucene` supertypes,
all the way up.

Behaviour can also change without any signature changing. Read the source of the Lucene methods that call
into cuvs-lucene and compare it with the previous release; the most important is
`AbstractKnnVectorQuery.rewrite`, which decides how each segment is searched. For example, Lucene 10.3
started expecting more than `k` hits from a filtered segment search, and throwing away a search that
returns fewer; `CompatKnnFloatVectorQuery.perLeafTopK` follows its rule, and has to follow it again if
it changes. The 10.3 variant also tells filtered segment searches from unfiltered ones by the visit limit
`rewrite` passes to `approximateSearch` (`Integer.MAX_VALUE` when it applies no filter), so check that
this still holds.

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
unzip -p ~/.m2/repository/org/apache/lucene/lucene-core/10.4.0/lucene-core-10.4.0.jar \
  META-INF/services/org.apache.lucene.codecs.Codec
```

If it is still `Lucene103Codec`, skip this step: the `Lucene103*` codecs keep writing on 10.4.
If it is new, for example `Lucene104Codec`:

1. Create `lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/` with the four codecs of the new
   generation, copied from `lucene-10.3/src/since/java` and renamed: `Lucene104AcceleratedHNSWCodec`,
   `Lucene104AcceleratedHNSWScalarQuantizedCodec`, `Lucene104AcceleratedHNSWBinaryQuantizedCodec`,
   `Lucene104CuVSGPUSearchCodec`. Each passes generation `104` and `() -> LuceneCompat.lucene104Codec()`
   (a lambda, not a method reference) to the `CuVSFilterCodec` constructor. The codec name
   is written into every segment, so never reuse or rename one once released.
2. Add `src/since/java` to the `add-since-sources` list in `lucene-10.4/pom.xml` (and, later, in every
   module after it).
3. In `lucene-10.4/.../LuceneCompat.java`:
   - add `lucene104Codec()` returning `new org.apache.lucene.codecs.lucene104.Lucene104Codec()`;
   - point `lucene103Codec()` at `org.apache.lucene.backward_codecs.lucene103.Lucene103Codec`, where
     Lucene moves the old default codec;
   - set `CURRENT_CODEC_GENERATION = 104`. The new codecs pass `104` as their generation to the
     `CuVSFilterCodec` constructor, and the `Lucene103*` codecs, which pass `103`, become read-only;
   - make `acceleratedHNSWCodec`, `acceleratedHNSWScalarQuantizedCodec`,
     `acceleratedHNSWBinaryQuantizedCodec` and `gpuSearchCodec` return the `Lucene104*` codecs.
4. Register the four new codecs in `lucene-10.4/src/main/resources/META-INF/services/org.apache.lucene.codecs.Codec`,
   after the existing ones. List only cuvs-lucene classes in the services files: a Lucene class there
   breaks every codec or format lookup in the JVM once a later release moves it.
5. Update the list of codecs per Lucene release in the class javadoc of `CuVSCodecs`.

If a vector format cuvs-lucene writes moved to `backward_codecs`, the codecs using it can no longer
write on the new release: add a new format next to the codecs of step 1, register it in
`META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat`, and have the new codecs and the matching
`LuceneCompat` format method use it.

### 5. Make the tests pass

```bash
mvn -pl lucene-10.4 -am verify
```

- New abstract methods or new tests in Lucene's `BaseKnnVectorsFormatTestCase` are handled once, in
  `src/test/java/.../BaseCuVSKnnVectorsFormatTestCase.java`, which every format test extends, and without
  `@Override`, so that they still compile against older releases.
- A test call whose signature changed goes through `TestLuceneCompat`, in a `compat/` variant (step 2).
- `TestBackCompat` fails if a codec from `CuVSCodecs` is not registered, or if an older codec can still
  write.

### 6. Run everything

```bash
./build.sh --run-java-tests --build-java-examples
```

This builds and tests every module, installs them, and builds the examples. It also checks the upgrade
from every earlier supported release: `TestBackCompatIndices` in each module writes indexes with its
codecs to `target/backcompat`, and reads and upgrades those that the modules for earlier releases wrote
there before it. Nothing needs to be committed for that.

### 7. Update the documentation

| File | Change |
| --- | --- |
| `README.md` (this directory) | the supported range and the artifact list; the codec paragraph if step 4 applied |
| `../../fern/pages/installation/java.md` | "supports Lucene 10.2 through 10.X", the artifact table |
| `../../fern/pages/user_guide/lucene.md` | the supported range, the artifact list, the codec table if step 4 applied |

The API reference pages in `fern/pages/lucene_api` are generated: the `fern-api-reference` pre-commit
hook picks up new classes in `lucene-*/src/since/java` by itself. Run `pre-commit run --all-files` last.

## After a cuvs-lucene release

Add the indexes written by the published jars to the back-compat tests, so that later changes cannot
silently make indexes written by that release unreadable. On a host with a GPU and the matching native
libraries:

```bash
src/test/backcompat/generate-released-indices.sh <released version>
```

This currently handles releases published as the single `cuvs-lucene` artifact (up to 26.10); releases
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
3. Keep its codecs and its `src/test/resources/backcompat/releases/*/10.X/` indexes as long as later
   releases must read indexes written with it.
4. Update the documentation listed in step 7 above.
5. When dropping 10.2, the relocation in `relocation/pom.xml` would point to an artifact that is no longer
   published: remove the `relocation` module (and stop publishing it) at that point. The deprecated
   `LuceneProvider` in `lucene-10.2/src/main/java`, kept only for callers of the old artifact, goes
   with the module.
