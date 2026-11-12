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
`jar-with-dependencies` assembly when present. Its uploaded build tree includes
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
released first. Each published module emits a completed-publication receipt.
The standalone dispatcher in release-tracking starts the registered docs project
in `rapidsai/docs`; docs failures do not prevent Lucene publication.
Branch and nightly builds keep their existing build behavior; these newly added
publication jobs run on release tags.

## Configuration and rollout

Use the existing `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, `MAVEN_DEPLOY_USERNAME`, and
`MAVEN_DEPLOY_TOKEN` conventions. The Sonatype token must authorize the Java and
Lucene namespaces, including every Lucene coordinate. The common staging profile defaults to `com.nvidia.cuvs`; set
`MAVEN_STAGING_PROFILE_ID` to the actual verified parent namespace if different,
for example `com.nvidia`. Confirm the namespace and account access during rollout.

App credentials remain in release-tracking's existing `release-dispatch`
environment. Deploy the standalone docs dispatcher, `cuvs-java` and `cuvs-lucene`
docs registrations, and the shared workflow receipt/output contract before
enabling this caller on the release branch. Only one system should publish these
coordinates for a release; retire the legacy manual publishing path at cutover.

The shared publisher prerequisite is
[shared-workflows #660](https://github.com/rapidsai/shared-workflows/pull/660).
Before it can support this caller, extend its single-POM discovery, metadata,
bundle validation, and Portal package-URL checks to accept the full Lucene
coordinate set and POM-only packaging. Sign and checksum the parent and relocation
POMs without requiring primary, sources, or javadoc JARs. Preserve the
`deployment-state: PUBLISHED` output and one completed receipt for the Lucene
release so documentation dispatch follows publication of the complete set.

Rollout order:

1. Extend and merge the shared publisher, including tests for mixed JAR and
   POM-only coordinates and resumed publication of that same deployment.
2. Deploy and verify the docs dispatcher and both docs registrations if docs
   automation is part of the cutover.
3. Confirm signing credentials, Sonatype namespace access, and staging profile;
   validate the full CUDA matrix's uploaded layout and signed bundle in the
   release environment.
4. Merge this caller and ensure it is present on the branch being tagged. Retire
   manual publishing and verify Java reaches `PUBLISHED` before the complete
   Lucene set is published. Check Maven resolution through the old relocation
   coordinates as well as every versioned coordinate.

The local checks exercise Maven assembly and docs configuration without compiling
Java or CUDA. The complete matrix build, Sonatype namespace authorization, App
dispatch, and S3 upload still require validation in the release environment.
