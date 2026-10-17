# Java and Lucene Maven releases

Release tags build both modules using the existing CUDA matrix. The release
gather job downloads each module's artifacts from the same workflow run, keeps
the CUDA builds in separate directories, and creates one Maven repository tree
per coordinate:

- `com.nvidia.cuvs:cuvs-java`
- `com.nvidia.cuvs.lucene:cuvs-lucene`

`assemble_maven_repo.py` takes the unclassified jar, sources, javadoc, and POM
from the highest numeric CUDA version in the matrix. This preserves the existing
higher-CUDA selection for common release artifacts and avoids merging
same-named jars from separate builds. Java retains each native architecture/CUDA
classifier. Lucene retains its attached `jar-with-dependencies` assembly when
present. Test jars and other Maven target directory contents are excluded.

Every input POM must match the release tag after normalizing a zero-padded patch
number. Lucene's POM must depend on the same Java version. Conflicting native
classifiers, missing required artifacts, invalid JARs, and unexpected build
directories fail gathering before publication.

Both modules use the shared Maven publisher with automatic publication enabled.
Lucene waits for the Java publisher's `PUBLISHED` output, so its dependency is
released first. Each published module emits a completed-publication receipt.
The standalone dispatcher in release-tracking starts the registered docs project
in `rapidsai/docs`; docs failures do not prevent Lucene publication.
Branch and nightly builds keep their existing build behavior; these newly added
publication jobs run on release tags.

## Configuration and rollout

Use the existing `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, `MAVEN_DEPLOY_USERNAME`, and
`MAVEN_DEPLOY_TOKEN` conventions. The Sonatype token must authorize both Maven
coordinates. The common staging profile defaults to `com.nvidia.cuvs`; set
`MAVEN_STAGING_PROFILE_ID` to the actual verified parent namespace if different,
for example `com.nvidia`. Confirm the namespace and account access during rollout.

App credentials remain in release-tracking's existing `release-dispatch`
environment. Deploy the standalone docs dispatcher, `cuvs-java` and `cuvs-lucene`
docs registrations, and the shared workflow receipt/output contract before
enabling this caller on the release branch. Only one system should publish these
coordinates for a release; retire the legacy manual publishing path at cutover.

The local checks exercise Maven assembly and docs configuration without compiling
Java or CUDA. The complete matrix build, Sonatype namespace authorization, App
dispatch, and S3 upload still require validation in the release environment.
