# Project Change Log

## Request: 2026-03-27 Exclude backend HR.md and .env.example

### Objective
- Add ignore rules so `backend/HR.md` and `backend/.env.example` are not uploaded.

### Status
- Completed.

### Changed Files
- Modified `.gitignore`
- Modified `PROJECT_CHANGE_LOG.md`

### Backup Created Before Edits
- `.gitignore` backup: `.codex-backups/20260327-ignore-backend-hr-envexample/.gitignore.bak`
- `PROJECT_CHANGE_LOG.md` backup: `.codex-backups/20260327-ignore-backend-hr-envexample/PROJECT_CHANGE_LOG.md.bak`

### Rollback Options
1. Git rollback (preferred):
   - `git restore .gitignore PROJECT_CHANGE_LOG.md`
2. Manual rollback from backup:
   - `Copy-Item '.codex-backups/20260327-ignore-backend-hr-envexample/.gitignore.bak' '.gitignore' -Force`
   - `Copy-Item '.codex-backups/20260327-ignore-backend-hr-envexample/PROJECT_CHANGE_LOG.md.bak' 'PROJECT_CHANGE_LOG.md' -Force`

### Verification Log
- Confirmed `.gitignore` now includes:
  - `backend/HR.md`
  - `backend/.env.example`
- Note: `git check-ignore` could not be run because this workspace currently has no `.git` metadata directory.

---

## Request: 2026-03-27 Exclude Tests and HR Walkthrough Files

### Objective
- Update `.gitignore` so the tests folder and HR walkthrough markdown files are not uploaded.

### Status
- Completed.

### Changed Files
- Modified `.gitignore`
- Modified `PROJECT_CHANGE_LOG.md`

### Backup Created Before Edits
- `.gitignore` backup: `.codex-backups/20260327-ignore-tests-walkthrough/.gitignore.bak`
- `PROJECT_CHANGE_LOG.md` backup: `.codex-backups/20260327-ignore-tests-walkthrough/PROJECT_CHANGE_LOG.md.bak`

### Rollback Options
1. Git rollback (preferred):
   - `git restore .gitignore PROJECT_CHANGE_LOG.md`
2. Manual rollback from backup:
   - `Copy-Item '.codex-backups/20260327-ignore-tests-walkthrough/.gitignore.bak' '.gitignore' -Force`
   - `Copy-Item '.codex-backups/20260327-ignore-tests-walkthrough/PROJECT_CHANGE_LOG.md.bak' 'PROJECT_CHANGE_LOG.md' -Force`

### Verification Log
- Confirmed `.gitignore` now includes:
  - `backend/tests/`
  - `tests/`
  - `backend/*WALKTHROUGH*.md`
- Note: `git check-ignore` could not be run because this workspace currently has no `.git` metadata directory.

---

## Request: 2026-03-27 Complete .gitignore for Current Repo Structure

### Objective
- Prepare a complete `.gitignore` tailored to this mixed Python backend + Vite React frontend repository.

### Status
- Completed.

### Changed Files
- Modified `.gitignore`
- Modified `PROJECT_CHANGE_LOG.md`

### Backup Created Before Edits
- `.gitignore` backup: `.codex-backups/20260327-gitignore-refresh/.gitignore.bak`
- `PROJECT_CHANGE_LOG.md` backup: `.codex-backups/20260327-gitignore-refresh/PROJECT_CHANGE_LOG.md.bak`

### Rollback Options
1. Git rollback (preferred):
   - `git restore .gitignore PROJECT_CHANGE_LOG.md`
2. Manual rollback from backup:
   - `Copy-Item '.codex-backups/20260327-gitignore-refresh/.gitignore.bak' '.gitignore' -Force`
   - `Copy-Item '.codex-backups/20260327-gitignore-refresh/PROJECT_CHANGE_LOG.md.bak' 'PROJECT_CHANGE_LOG.md' -Force`

### Verification Log
- Manual rule verification completed by checking `.gitignore` entries for:
  - `Frontend/node_modules`, `Frontend/dist`
  - Python caches (`__pycache__`, `*.py[cod]`)
  - env files (`.env*`, `Frontend/.env*`, `backend/.env*`)
- Note: `git check-ignore` could not be run because this workspace currently has no `.git` metadata directory.

---

## Request: 2026-03-27 Frontend React Workspace

### Objective
- Build a separate `Frontend` React + JavaScript UI for HR screening without breaking existing backend behavior.

### Status
- Completed.

### Changed Files
- Added `Frontend/package.json`
- Added `Frontend/vite.config.js`
- Added `Frontend/index.html`
- Added `Frontend/package-lock.json`
- Added `Frontend/.env.example`
- Added `Frontend/README.md`
- Added `Frontend/src/main.jsx`
- Added `Frontend/src/api.js`
- Added `Frontend/src/App.jsx`
- Added `Frontend/src/styles.css`
- Added `PROJECT_CHANGE_LOG.md`
- Modified `.gitignore`

### Backup Created Before Edits
- `.gitignore` backup: `.codex-backups/20260327-frontend-ui/.gitignore.bak`

### Rollback Options
1. Git rollback (preferred):
   - `git restore .gitignore`
   - `git clean -fd Frontend`
   - `git restore PROJECT_CHANGE_LOG.md`
2. Manual rollback for `.gitignore`:
   - `Copy-Item '.codex-backups/20260327-frontend-ui/.gitignore.bak' '.gitignore' -Force`
   - Remove added frontend folder: `Remove-Item -LiteralPath 'Frontend' -Recurse -Force`

### Verification Log
- `python -m unittest discover -s tests` (in `backend`): pass (14 tests, OK)
- `npm run build` (in `Frontend`): pass
- Notes:
  - `python -m pytest` was unavailable because `pytest` is not installed in this environment.

---

## Request: 2026-03-27 Remove Backend Status Header Block

### Objective
- Remove the "Backend online / Model ready" section from the frontend header.

### Status
- Completed.

### Changed Files
- Modified `Frontend/src/App.jsx`
- Modified `Frontend/src/styles.css`
- Modified `PROJECT_CHANGE_LOG.md`

### Rollback Options
1. Git rollback (preferred):
   - `git restore Frontend/src/App.jsx Frontend/src/styles.css PROJECT_CHANGE_LOG.md`
2. Manual rollback:
   - Restore the removed header block and related health styles/effects from the previous commit or local history.

### Verification Log
- `npm run build` (in `Frontend`): pass

---

## Request: 2026-03-30 ATS Semantic + Reasoning Upgrade

### Objective
- Upgrade lexical ATS scoring to semantic/context-aware scoring with reasoning and ranking tie-breakers, while preserving API compatibility.

### Status
- Completed.

### Changed Files
- Modified `backend/app/agent.py`
- Modified `backend/app/main.py`
- Modified `backend/app/schemas.py`
- Modified `backend/app/scoring.py`
- Modified `backend/requirements.txt`
- Added `backend/app/semantic_matching.py`
- Added `backend/app/reasoning.py`
- Added `backend/app/ranking.py`
- Modified `backend/tests/test_api.py`
- Modified `backend/tests/test_scoring.py`
- Added `backend/tests/test_semantic_matching.py`
- Added `backend/tests/test_reasoning.py`
- Added `backend/tests/test_ranking.py`
- Modified `PROJECT_CHANGE_LOG.md`

### Backup Created Before Log Update
- `PROJECT_CHANGE_LOG.md` backup: `.codex-backups/20260330-ats-upgrade/PROJECT_CHANGE_LOG.md.bak`

### Rollback Options
1. Git rollback (preferred):
   - `git restore backend/app/agent.py backend/app/main.py backend/app/schemas.py backend/app/scoring.py backend/requirements.txt backend/tests/test_api.py backend/tests/test_scoring.py PROJECT_CHANGE_LOG.md`
   - `git clean -fd backend/app/semantic_matching.py backend/app/reasoning.py backend/app/ranking.py backend/tests/test_semantic_matching.py backend/tests/test_reasoning.py backend/tests/test_ranking.py`
2. Manual rollback for log file only:
   - `Copy-Item '.codex-backups/20260330-ats-upgrade/PROJECT_CHANGE_LOG.md.bak' 'PROJECT_CHANGE_LOG.md' -Force`

### Verification Log
- `python -m unittest discover -s tests` (in `backend`): pass (`23` tests, `OK`)

---

## Request: 2026-03-30 Dynamic Skill Intelligence Upgrade

### Objective
- Replace hardcoded skill extraction with dynamic extraction.
- Add semantic utilities for batched skill matching, additional relevant skill detection, clustering, and domain tag detection.
- Upgrade scoring to use semantic utilities, additional-skill bonus, and domain boost while keeping API compatibility.

### Status
- Completed.

### Changed Files
- Modified `backend/app/agent.py`
- Modified `backend/app/semantic_matching.py`
- Modified `backend/app/scoring.py`
- Modified `backend/app/schemas.py`
- Modified `backend/tests/test_api.py`
- Modified `backend/tests/test_fallbacks.py`
- Modified `backend/tests/test_scoring.py`
- Modified `backend/tests/test_semantic_matching.py`
- Modified `PROJECT_CHANGE_LOG.md`

### Backup Created Before Edits
- Backup root: `.codex-backups/20260330-131913-dynamic-skill-intelligence`
- Included backups:
  - `backend/app/agent.py`
  - `backend/app/semantic_matching.py`
  - `backend/app/scoring.py`
  - `backend/app/schemas.py`
  - `backend/tests/test_api.py`
  - `backend/tests/test_fallbacks.py`
  - `backend/tests/test_scoring.py`
  - `backend/tests/test_semantic_matching.py`
  - `PROJECT_CHANGE_LOG.md`

### Rollback Options
1. Git rollback for tracked files (preferred):
   - `git restore backend/app/agent.py backend/app/scoring.py backend/app/schemas.py backend/tests/test_api.py backend/tests/test_fallbacks.py backend/tests/test_scoring.py backend/tests/test_semantic_matching.py PROJECT_CHANGE_LOG.md`
2. Manual rollback from backup (works for both tracked and untracked files):
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/app/agent.py' 'backend/app/agent.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/app/semantic_matching.py' 'backend/app/semantic_matching.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/app/scoring.py' 'backend/app/scoring.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/app/schemas.py' 'backend/app/schemas.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/tests/test_api.py' 'backend/tests/test_api.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/tests/test_fallbacks.py' 'backend/tests/test_fallbacks.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/tests/test_scoring.py' 'backend/tests/test_scoring.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/backend/tests/test_semantic_matching.py' 'backend/tests/test_semantic_matching.py' -Force`
   - `Copy-Item '.codex-backups/20260330-131913-dynamic-skill-intelligence/PROJECT_CHANGE_LOG.md' 'PROJECT_CHANGE_LOG.md' -Force`

### Verification Log
- `python -m unittest discover -s tests` (in `backend`): pass (`31` tests, `OK`)
