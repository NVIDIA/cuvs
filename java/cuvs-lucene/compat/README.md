# Lucene API variants

Some Lucene classes that cuvs-lucene extends or calls changed between Lucene releases. The code adapting
to such a class differs between releases, but is the same in all the releases that share one version of
the class, so it lives here once instead of being copied into every `lucene-10.X` module.

Each directory is one **variant**: the adapter for one Lucene class, as that class looks from one release
on. Directories are named `<class>-<release>-<change>`:

- `<class>` is the Lucene class being adapted. There is one `cuvs.lucene.compat.<class>` property per
  class, and each module's `pom.xml` sets it to the variant its Lucene release needs. The parent `pom.xml`
  adds the chosen directories to the module's sources.
- `<release>` is the first supported Lucene release with that version of the class (`102` is the oldest
  supported release, so `102` variants describe how the class looked before any change we adapt to).
  Names never change afterwards, even when older releases are dropped.
- `<change>` says what that release changed, so that variants can be told apart without opening them.

| Property | Variant | Used by | What changed |
| --- | --- | --- | --- |
| `cuvs.lucene.compat.knnVectorsReader` | `knn-vectors-reader-102-bits` | 10.2 | `search` takes the accepted documents as `Bits` |
| | `knn-vectors-reader-103-acceptdocs` | 10.3–10.5 | `search` takes `AcceptDocs`; readers also report `getOffHeapByteSize` |
| `cuvs.lucene.compat.knnVectorsWriter` | `knn-vectors-writer-102-void-merge` | 10.2–10.4 | `mergeOneField` returns `void` |
| | `knn-vectors-writer-105-deferred-merge` | 10.5 | `mergeOneField` returns deferred work as an `IORunnable` |
| `cuvs.lucene.compat.knnFloatVectorQuery` | `knn-float-vector-query-102-bits` | 10.2 | `approximateSearch` takes `Bits` |
| | `knn-float-vector-query-103-acceptdocs` | 10.3–10.5 | `approximateSearch` takes `AcceptDocs` |
| `cuvs.lucene.compat.leafReader` (tests only) | `leaf-reader-102-bits` | 10.2 | `searchNearestVectors` takes `Bits` |
| | `leaf-reader-103-acceptdocs` | 10.3–10.5 | `searchNearestVectors` takes `AcceptDocs` |

Code that differs in (almost) every release, such as which Lucene classes moved to `backward_codecs`,
stays in each module's own `LuceneCompat`.

- **A new release changes a class that already has variants:** copy the variant the previous release uses
  to `<class>-<new release>-<change>`, change it there, and point the new module's property at it. Never
  edit a variant that another module still uses.
- **A new release changes a class with no variants yet:** move the adapter out of the shared `src/` into
  two variants, `<class>-102-<old shape>` and `<class>-<new release>-<change>`, add a
  `cuvs.lucene.compat.<class>` property to every module, and a matching `<source>` line to the
  `add-shared-sources` (or `add-shared-test-sources`) execution of the parent `pom.xml`.
- **A release is dropped:** delete every variant no remaining module uses. When all remaining modules use
  the same variant of a class, move its code into the shared `src/`, delete the variant and the property,
  and remove its `<source>` line from the parent `pom.xml`.
