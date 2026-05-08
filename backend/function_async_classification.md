# Codebase Asynchronous Verification Matrix

This walkthrough categorizes every function across the application components based on the concurrency characteristics you requested. 

The categories represent:
1. **Async**: Natively uses `async def`.
2. **Sync**: Natively uses `def` (Executing synchronously on the main thread).
3. **Can be converted to Async**: Functions currently synchronous but that perform **I/O bound operations** (Network requests to LLMs, File I/O, Subprocesses) where yielding to the event loop is heavily beneficial without requiring threadpools.
4. **Cannot/Should not be converted to Async**: Functions performing pure **CPU/GPU-bound operations** (Regex text parsing, mathematical scoring, local ML model inference, Pydantic validators). Forcing these into `async def` provides no system throughput benefit and simply blocks the event loop unless offloaded to a ThreadPoolExecutor.

---

## 1. `main.py` (API Routing & Orchestration)
| Function Name | Current State | Target Async State | Reasoning |
| :--- | :--- | :--- | :--- |
| `startup_event` | **Async** | - | Application startup lifecycle. |
| `screen_resumes` | **Async** | - | Concurrent orchestration and request ingestion. |
| `process_single_resume` | **Async** | - | Gated concurrency operation. |
| `root` | **Sync** | **Can be converted** | FastAPI allows async route handlers. Very trivial, but network native. |
| `health` | **Sync** | **Can be converted** | FastAPI allows async route handlers. |
| `_sync_process` | **Sync** | **Can be converted** | Wrapping function. Parts of this pipeline (LLM calls) can be awaited. Currently, it is completely offloaded to a thread pool via `asyncio.to_thread`. |

## 2. `llm_understanding.py` (Generative AI Integration)
*This module communicates with external APIs (Groq, Gemini). Almost all functions here should ideally be async to prevent blocking.*
| Function Name | Current State | Target Async State | Reasoning |
| :--- | :--- | :--- | :--- |
| `_generate_json` | **Sync** | **Can be converted** | Heavy Network I/O. Should use `AsyncGroq` and Gemini `generate_content_async`. |
| `enrich_jd_intelligence` | **Sync** | **Can be converted** | Wraps `_generate_json`. |
| `generate_skill_graph` | **Sync** | **Can be converted** | Wraps `_generate_json`. |
| `enrich_candidate_persona` | **Sync** | **Can be converted** | Wraps `_generate_json`. |
| `extract_json_from_text` | **Sync** | **Can be converted** | Wraps `_generate_json`. |
| `_load_cache` | **Sync** | **Can be converted** | Disk I/O. Could use `aiofiles`. |
| `_save_cache` | **Sync** | **Can be converted** | Disk I/O. Could use `aiofiles`. |

## 3. `agent.py` (Agentic Pipeline & Extraction)
| Function Name | Current State | Target Async State | Reasoning |
| :--- | :--- | :--- | :--- |
| `extract_job_description` | **Sync** | **Can be converted** | Calls LLM fallbacks (Network I/O). |
| `extract_resume` | **Sync** | **Can be converted** | Calls LLM fallbacks (Network I/O). |
| `_extract_structured_output` | **Sync** | **Can be converted** | Controls the retry logic involving Generative AI APIs. |
| `_load_model` / `preload` | **Sync** | **Cannot be converted** | VLM Loading (PyTorch) strictly blocks. |
| `generate_response` | **Sync** | **Cannot be converted** | `model.generate()` is a pure GPU computation block. HuggingFace Transformers do not yield to asyncio natively. |
| `has_meaningful_extraction` | **Sync** | **Cannot be converted** | Pure CPU dictionary evaluation. |
| `parse_json`, `_repair_truncated_json` | **Sync** | **Cannot be converted** | Regex and string processing logic. |
| `job_description_prompt`, `resume_prompt` | **Sync** | **Cannot be converted** | String formatting limits. |
| `heuristic_*` (All heuristic rules) | **Sync** | **Cannot be converted** | Pure regex mapping and text tokenization. Completely CPU bound. |

## 4. `document_parser.py` & `ocr.py` (Document Processing)
| Function Name | Current State | Target Async State | Reasoning |
| :--- | :--- | :--- | :--- |
| `extract_with_boxes` (`ocr.py`) | **Sync** | **Can be converted** | `pytesseract` spawns a subprocess via `Popen`. This could be rewritten to use `asyncio.create_subprocess_exec` to free the thread while Tesseract works. |
| `_parse_pdf` (`document.py`) | **Sync** | **Can be converted** | `pdf2image` internally spawns poppler binaries. Can be converted similar to tesseract. |
| `_parse_docx` / `_parse_image` | **Sync** | **Cannot be converted** | Loading PIL arrays or querying docx files locally is CPU bound XML/Array processing. |
| `clean_ocr_text`, `annotate_*` | **Sync** | **Cannot be converted** | Regex array filtering. CPU bound memory processing. |

## 5. `reasoning.py` (Logical Synthesis)
| Function Name | Current State | Target Async State | Reasoning |
| :--- | :--- | :--- | :--- |
| `generate_reasoning` | **Sync** | **Can be converted** | Transmits the scoring payload to LLMs (Network I/O). |
| `fallback_reasoning` | **Sync** | **Cannot be converted** | In-memory string evaluation. |
| `build_reasoning_prompt` | **Sync** | **Cannot be converted** | Pure string manipulation logic. |

## 6. `scoring.py`, `ranking.py`, `schemas.py`, & `preprocess.py` (The Mathematical Core)
*These modules represent the backbone mathematical compute of the application. They analyze strings, multiply matrices, compute cosine similarities, filter arrays natively, and validate static data schemas.*

**All functions in these 4 files are Sync and Cannot (and should not) be converted into async.**
* **Examples (`scoring.py`)**: `score_candidate`, `semantic_coverage`, `cluster_skill_evidence`, `progression_score`. These functions compute continuous mathematical heuristics directly on CPU logic.
* **Examples (`semantic_matching.py`)**: `best_match`, `match_skills`, `lexical_similarity`. Uses `SentenceTransformer` which encodes vectors synchronously on local compute.
* **Examples (`preprocess.py`)**: `preprocess`. Executes pure OpenCV NumPy array convolutions (`cv2.medianBlur`, Image Matrix Math). Natively blocking.
* **Examples (`schemas.py`)**: Pydantic `@field_validator` and `@model_validator` functions dynamically parse dates and enforce structure using Python internals. Native Pydantic is entirely synchronous.
* **Examples (`ranking.py`)**: Uses Python's native `sorted()` map algorithm. Completely synchronous.

> [!CAUTION]
> If these mathematical, regex, and ML processing sections natively halted as `async def`, they would actually **lock** the entire FastAPI event loop leading to disastrous bottlenecking. Keeping them `Sync` and offloading the heavy wrappers to pool threads (via `asyncio.to_thread` like what is currently done in `main.py#152`) is the standard production architectural pattern!
