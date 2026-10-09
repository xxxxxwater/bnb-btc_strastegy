# Publishing process

The first reconstructed baseline is v0.3.0. A main-branch commit with first-line message 'release: v0.3.0' starts the publish workflow after source changes have been tested. The workflow validates the version, runs unit tests and Python compilation, builds/pushes the Docker image to GHCR, creates a Git tag, then creates a GitHub Release with RELEASE_NOTES.md.

Source: https://github.com/xxxxxwater/bnb-btc_strastegy/releases

Image (only after CI publisher success): ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.3.0

~~~bash
docker pull ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.3.0
~~~

Do not treat workflow configuration as proof of successful publication. Verify green workflow, actual Release, and visible GHCR package. The GITHUB_TOKEN must have contents:write and packages:write; host settings can still block publishing. The image contains no credentials, and running it requires external strategy config. Mutable base image must be digest-pinned in a production deployment. No private V4 model is shipped and no performance guarantees apply.
