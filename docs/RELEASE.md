# Publishing process

The first reconstructed baseline is v0.3.0. A main-branch commit with first-line message 'release: v0.3.0' starts the publish workflow after source changes have been tested. The workflow validates the version, runs unit tests and Python compilation, builds/pushes the Docker image to GHCR, creates a Git tag, then creates a GitHub Release with RELEASE_NOTES.md.

Source: https://github.com/xxxxxwater/bnb-btc_strastegy/releases

Image (only after CI publisher success): ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.3.0

~~~bash
docker pull ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.3.0
~~~

Do not treat workflow configuration as proof of successful publication. Verify green workflow, actual Release, and visible GHCR package. The GITHUB_TOKEN must have contents:write and packages:write; host settings can still block publishing. The image contains no credentials, and running it requires external strategy config. Mutable base image must be digest-pinned in a production deployment. No private V4 model is shipped and no performance guarantees apply.


## v0.4.0 MLSD publish

The v0.4.0 source integrates MLSD v0.2.1 from its SHA-pinned upstream source archive. The release workflow installs the MLSD extra during tests and Docker build, and only creates v0.4.0 GitHub Release after the GHCR image push succeeds. The v0.3.0 versioned image and GitHub release are retained. Verify publication before pulling ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.4.0.


## v0.4.1

No change to existing v0.4.0 tags. v0.4.1 validates exact timestamp alignment between research and trading, adds Ruff correctness checks, model v2 provenance validation, and verifies the real Freqtrade class imports during GHCR Docker build. Only call the Release and package published once GitHub Actions reports both image push and release creation successful. The source/default configuration remains dry-run with execution stopped.
