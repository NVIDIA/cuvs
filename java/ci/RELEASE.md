# Java and Lucene Maven releases

Release tags build both modules using the existing CUDA matrix. The release
gather job downloads each module's artifacts from the same workflow run, keeps
the CUDA builds in separate directories, and creates one Maven repository tree
per coordinate:

- `com.nvidia.cuvs:cuvs-java`
- `com.nvidia.cuvs.lucene:cuvs-lucene-10.X` for every module registered in the
  release source's `java/cuvs-lucene/pom.xml` (currently 10.2 through 10.6)
- `com.nvidia.cuvs.lucene:cuvs-lucene-parent` (POM only)
- `com.nvidia.cuvs.lucene:cuvs-lucene` (POM-only relocation to `cuvs-lucene-10.2`)

`assemble_maven_repo.py` takes the unclassified jar, sources, javadoc, and POM
from the highest numeric CUDA version in the matrix. This preserves the existing
higher-CUDA selection for common release artifacts and avoids merging
same-named jars from separate builds. Java retains each native architecture/CUDA
classifier. Lucene retains each versioned module's attached
`jar-with-dependencies` assembly. Its uploaded build tree includes
the reactor's `target/` and every child module's `target/`, preserving their
separate directories. Flattened POMs supply standalone coordinates and resolved
Java dependencies. POM-only artifacts require no JARs. Test jars and other Maven
target directory contents are excluded.

Every input POM must match the release tag after normalizing a zero-padded patch
number. Lucene's POM must depend on the same Java version. Conflicting native
classifiers, missing required artifacts, invalid JARs, and unexpected build
directories fail gathering before publication. Every CUDA build must contain the
complete Lucene reactor; the relocation must point to a coordinate in this
release. The highest-CUDA build supplies the entire Lucene publication set.

Both modules use the shared Maven publisher with automatic publication enabled.
Lucene waits for the Java publisher's `PUBLISHED` output, so its dependency is
released first. Javadoc JARs remain part of the Maven publications. Website
documentation for Java and Lucene is part of the existing cuVS Fern docs.
Branch and nightly builds keep their existing build behavior; these newly added
publication jobs run on release tags.

## Configuration and rollout

Use the existing `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, `MAVEN_DEPLOY_USERNAME`, and
`MAVEN_DEPLOY_TOKEN` conventions. The Sonatype token must authorize the Java and
Lucene namespaces, including every Lucene coordinate. The Central Publisher Portal
API checks account namespace access directly. The legacy
`MAVEN_STAGING_PROFILE_ID` setting is accepted but unused. Confirm namespace and
account access during rollout.

Only one system should publish these coordinates for a release; retire the
legacy manual publishing path at cutover.

The shared publisher prerequisite is
[shared-workflows #660](https://github.com/rapidsai/shared-workflows/pull/660).
It adds a coordinate manifest, POM-only packaging, full Portal package-URL
verification, and recovery tied to the complete publication set. The caller sets
`primary-artifact-id: cuvs-lucene` to select the primary workflow output coordinate.
The `deployment-state: PUBLISHED` output covers the entire Lucene set.

Rollout order:

1. Merge the shared publisher, including its mixed JAR/POM-only coordinate tests
   and resumed-publication verification.
2. Confirm signing credentials and Sonatype namespace access;
   validate the full CUDA matrix's uploaded layout and signed bundle in the
   release environment.
3. Merge this caller and ensure it is present on the branch being tagged. Retire
   manual publishing and verify Java reaches `PUBLISHED` before the complete
   Lucene set is published. Check Maven resolution through the old relocation
   coordinates as well as every versioned coordinate.

Run `pre-commit run test-maven-assembly --all-files` to check Maven assembly
without building Java or CUDA. The PR checks run the same hook.

The complete matrix build and Sonatype namespace authorization still require
validation in the release environment. No separate Java docs dispatcher or
Javadoc website upload is required for this Maven release path.
