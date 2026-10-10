# Docker compatibility refresh after the v0.4.1 Release

The published **GitHub Release v0.4.1 tag is immutable**. Additional commits on `main`
are not part of that original source snapshot. For compatibility with operators using
`docker pull ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.4.1`, a separate guarded
workflow deliberately repushes the **mutable GHCR tag** when a main commit's first
line is exactly `build: refresh v0.4.1 GHCR`.

## Safeguards

- Require the original GitHub release to exist and the project version to remain 0.4.1.
- Run Python tests, Ruff, compileall and package integrity checks before image build.
- Docker build validates the real MLSD import and Freqtrade strategy under ftuser.
- Publish `:v0.4.1`, `:latest` and `:sha-<full commit SHA>`; do not move Git tags.
- OCI `org.opencontainers.image.revision` records the source commit.
- Record the image digest after the build. Pin an exact digest in deployments.

**Warning:** repushing `:v0.4.1` breaks tag immutability. An image pulled before
this refresh may differ from one pulled later. The original GitHub Release
v0.4.1 remains at its historical source commit. This is a user-requested
compatibility behavior, not recommended for normal semantic releases.

## Pull

```bash
docker pull ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.4.1
docker image inspect ghcr.io/xxxxxwater/bnb-btc_strastegy:v0.4.1
```

Use the printed GHCR digest and image revision label for reproducibility.
Pulling an image does not deploy or activate a trading bot; the example config
remains `dry_run=true` and `initial_state=stopped`.
