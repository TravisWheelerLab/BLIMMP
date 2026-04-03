# Developer guide

This guide covers the repository layout, how to make changes, and how to release a new version including pushing a new container image to the GitHub Container Registry.

---

## Repository layout

```
BLIMMP/
├── BLIMMP_Scripts/
│   ├── module_detection.py       # Main entry point — BLIMMP algorithm
│   ├── Data_Dependencies/        # ko_list.txt, module_freq.txt, taxonomy files
│   └── Graph_Dependencies/       # KEGG module graphs and equation data
├── Examples/                     # Example input/output files
├── Dockerfile                    # Container definition (see below)
├── setup.py                      # Package definition and entry points
├── requirements.txt              # Python dependencies
└── .github/
    └── workflows/
        └── docker.yml            # GitHub Actions: build and push container on tag
```

The package is installed via `pip install .`. The `BLIMMP` command (defined in `setup.py` as a `console_scripts` entry point) calls `BLIMMP_Scripts.module_detection:main`.

---

## Making changes

Work on a feature branch and open a PR against `main`:

```bash
git checkout -b my-feature
# ... make changes ...
git add -p
git commit -m "describe the change"
git push -u origin my-feature
# open PR on GitHub
```

---

## Releasing a new version

Releases are triggered by pushing a version tag. The GitHub Actions workflow (`.github/workflows/docker.yml`) automatically builds the Docker container and pushes it to the GitHub Container Registry (`ghcr.io`) when a tag matching `v*` is pushed.

### Step-by-step

1. **Merge your PR** to `main`.

2. **Update the version** in `setup.py`:
   ```python
   version="1.2.3",
   ```

3. **Commit the version bump** directly on `main` (or via a PR):
   ```bash
   git checkout main && git pull
   git add setup.py
   git commit -m "Bump version to 1.2.3"
   git push
   ```

4. **Tag the release** — this triggers the container build:
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

5. **Verify the build** at `https://github.com/TravisWheelerLab/BLIMMP/actions`. The workflow pushes three tags to `ghcr.io/traviswheelerlab/blimmp`:
   - `1.2.3` (full version — use this in production)
   - `1.2` (minor version)
   - `latest`

6. **Update blimmp-nf** — edit the `container` directive in the `RUN_BLIMMP` process in `blimmp-nf/main.nf`:
   ```groovy
   container 'ghcr.io/traviswheelerlab/blimmp:1.2.3'
   ```
   Commit and push to blimmp-nf.

---

## How the GitHub Actions workflow works

The workflow file is `.github/workflows/docker.yml`. It fires on any tag push matching `v*` and:

1. Checks out the repo
2. Logs in to `ghcr.io` using the `GITHUB_TOKEN` secret (automatically provided by GitHub Actions — no manual secret setup required)
3. Extracts version tags from the Git tag using `docker/metadata-action`
4. Builds the image from `Dockerfile` and pushes all three tags

### Requirements for the push to succeed

- The GitHub Actions workflow must have `packages: write` permission. This is set in `docker.yml` under `permissions:` and does not require any manual configuration.
- The container package on `ghcr.io` must be set to **public** (or the runner must have access). The first time a tag is pushed, GitHub creates the package automatically. To make it public: go to `https://github.com/TravisWheelerLab/BLIMMP/pkgs/container/blimmp` → Package settings → Change visibility → Public.
- The tag must match `v*` (e.g. `v1.2.3`). Tags without the `v` prefix will not trigger the workflow.

---

## The Dockerfile

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
```

- `python:3.10-slim` — minimal Python 3.10 base image
- `procps` — provides `ps`, required by Nextflow to collect task metrics
- `pip install .` — installs BLIMMP and all dependencies from `setup.py`/`requirements.txt`

If you add a new system-level dependency (e.g. a C library), add it to the `apt-get install` line. If you add a new Python dependency, add it to `setup.py` under `install_requires` (and optionally `requirements.txt`).

---

## Testing the container locally

If you have Docker installed:

```bash
docker build -t blimmp-test .
docker run --rm blimmp-test BLIMMP -h
```

On an HPC cluster with Singularity (after pushing to `ghcr.io`):

```bash
singularity exec docker://ghcr.io/traviswheelerlab/blimmp:1.2.3 BLIMMP -h
```
