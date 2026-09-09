# Dataset notes

`experiments/argus.yaml` and the cluster scripts refer to meshes under
`experiments/data/` (historically `data/synthetic/*.obj` and `data/real/*.obj`).
Those full inputs are **not** shipped in this repository.

## Options (pick one before a public paper release)

1. **Tracked fixtures** — `data/fixtures/torus.obj` is a small redistributable
   mesh used by `experiments/smoke.yaml` for CPU smoke tests.
2. **External dataset path** — place or symlink the full research meshes at
   `experiments/data/` (or set absolute paths in your config). Keep large and
   restricted meshes out of git.
3. **Download + checksums** — when a redistributable pack exists, document the
   URL and SHA256 checksums here.

## Smoke fixture

| File | Role |
| --- | --- |
| `data/fixtures/torus.obj` | Minimal torus for import and short optimization smokes |

Do not commit generated checkpoints, WandB state, paper renderings, or large
derived meshes.
