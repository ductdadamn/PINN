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

## Handoffs

When a task is completed, a `handoffs/handoff_<name>.md` file is generated documenting what was built, how to run it, dependencies, and integration notes. Check `handoffs/` before starting work that depends on a teammate's task.

## Rules

- Do not alter core mathematical equations or loss functions without explicit instruction from the Strategic Planner / lead.
- Keep `data/`, `outputs/`, and `.venv/` out of git (see `.gitignore`) — never commit raw data or model artifacts.
- Branch names must follow `feature/<name>/<task-slug>` (see examples above) — no unrelated text, no inappropriate language.
- **No solo resolution of shared-logic merge conflicts in the GitHub web UI.** If a conflict touches `src/losses.py`, `src/data_loader.py`, or a model's `shared_parameters()`/`forward()` contract, ping the lead before resolving it — a wrong pick between "current" and "incoming" is easy to make and easy to miss, and has broken `develop` for everyone at least once already (see `handoffs/sprint2_final_report.md`, §"Repo hygiene notes"). Trivial conflicts (e.g. a docstring wording difference) are fine to resolve solo.
