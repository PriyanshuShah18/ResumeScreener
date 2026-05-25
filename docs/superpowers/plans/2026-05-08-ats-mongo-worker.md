# ATS Mongo Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone worker that scans MongoDB job applications, scores candidates against jobs, and writes ATS score logs.

**Architecture:** Keep the existing FastAPI upload endpoint unchanged. Add focused modules for ATS settings, MongoDB persistence, S3-compatible resume download, HRMS document-to-schema conversion, existing scoring-engine orchestration, and a polling worker entry point. Resume parsing continues to use `DocumentParser` and `HRExtractionService`; scoring continues to use `score_candidate()` from `app/scoring.py`.

**Tech Stack:** Python 3.11, existing FastAPI/Pydantic screening engine, PyMongo, boto3, pytest/unittest.

---

### Task 1: ATS Settings

**Files:**
- Create: `backend/app/ats_settings.py`
- Test: `backend/tests/test_ats_settings.py`

- [ ] Write tests for env parsing defaults and required S3/Mongo settings.
- [ ] Add `ATSSettings` dataclass and `get_ats_settings()`.
- [ ] Verify tests pass with `python -m pytest backend/tests/test_ats_settings.py -q`.

### Task 2: S3 Resume Storage Adapter

**Files:**
- Create: `backend/app/storage.py`
- Test: `backend/tests/test_storage.py`

- [ ] Write tests for downloading by `storageKey` and preserving filename/mime type.
- [ ] Add a `StoredResume` dataclass and `S3ResumeStorage.download_resume()`.
- [ ] Verify tests pass with `python -m pytest backend/tests/test_storage.py -q`.

### Task 3: HRMS Worker Mapping

**Files:**
- Modify: `backend/app/ats_worker.py`
- Test: `backend/tests/test_ats_worker.py`

- [ ] Write tests proving the worker converts HRMS job documents to `JobDescriptionData`.
- [ ] Write tests proving the worker converts candidate parsed JSON/profile fallback to `ResumeData`.
- [ ] Write tests proving the worker delegates directly to the existing `score_candidate()` engine from `app/scoring.py`.
- [ ] Keep mapping/log formatting in the worker and do not create a separate ATS scoring module.
- [ ] Verify tests pass with `python -m unittest discover -s tests -p "test_ats_*.py"`.

### Task 4: Mongo Repository

**Files:**
- Create: `backend/app/ats_mongo.py`
- Test: `backend/tests/test_ats_mongo.py`

- [ ] Write tests with fake collections for pending application filtering and log insertion shape.
- [ ] Implement `ATSMongoRepository`.
- [ ] Verify tests pass with `python -m pytest backend/tests/test_ats_mongo.py -q`.

### Task 5: Worker Orchestration

**Files:**
- Create: `backend/app/ats_worker.py`
- Test: `backend/tests/test_ats_worker.py`

- [ ] Write tests for using `parsedJson`, downloading/parsing from S3 when missing, inserting logs, and optional shortlist status update.
- [ ] Implement `ATSWorker.process_once()` and CLI loop.
- [ ] Verify tests pass with `python -m pytest backend/tests/test_ats_worker.py -q`.

### Task 6: Dependencies And Deployment

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`

- [ ] Add `pymongo` and `boto3`.
- [ ] Document MongoDB/S3/worker env vars in `.env.example`.
- [ ] Run focused ATS tests and a broader backend test pass.
