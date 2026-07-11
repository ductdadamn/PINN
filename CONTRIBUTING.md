# Contributing Workflow

Team: leader (Lead Execution Developer, working with an AI-assisted coding agent) + Kieu, Kem, Cam.

## Branching model (GitFlow-lite)

- `main` — always stable. Only updated via PR from `develop` at milestones. **Protected**: no direct pushes, PR + 1 review required.
- `develop` — integration branch. All feature work merges here first via PR.
- `feature/<name>/<task-slug>` — one branch per task, per person. Branch off `develop`, PR back into `develop`.

Examples:
- `feature/kieu/data-preprocessing`
- `feature/kem/loss-function-ablation`
- `feature/cam/inference-benchmark`

## Workflow

1. Pull latest `develop`: `git checkout develop && git pull origin develop`
2. Create your task branch: `git checkout -b feature/<name>/<task-slug>`
3. Commit, push: `git push -u origin feature/<name>/<task-slug>`
4. Open a PR into `develop` on GitHub.
5. After review/merge, delete the feature branch.
6. Periodically (at milestones), the lead opens a PR from `develop` into `main`.

## Milestone results (`results/`)

`outputs/` is gitignored (scratch/ephemeral — regenerate by re-running scripts, never commit raw data or routine checkpoints there). For an actual sprint milestone deliverable (a key trained checkpoint + its proof plot), commit a small curated copy to `results/<milestone-name>/` instead — `.gitignore` has an explicit exception for this directory. Include a short `README.md` in that folder with the headline metrics and the exact command to reproduce it. If the result depends on code that gets changed later (e.g. a bugfix that alters training behavior), tag the commit it was produced on (`git tag <name> <commit>`) so it stays reproducible — see `sprint2-fcn-baseline` for an example.

## Handoffs

When a task is completed, a `handoffs/handoff_<name>.md` file is generated documenting what was built, how to run it, dependencies, and integration notes. Check `handoffs/` before starting work that depends on a teammate's task.

## Rules

- Do not alter core mathematical equations or loss functions without explicit instruction from the Strategic Planner / lead.
- Keep `data/`, `outputs/`, and `.venv/` out of git (see `.gitignore`) — never commit raw data or model artifacts.
