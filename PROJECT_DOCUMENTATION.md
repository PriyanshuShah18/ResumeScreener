# HRScreener - Project Documentation

**An Executive Summary**
HRScreener is an intelligent, automated resume screening system built to alleviate the manual burden placed on recruiters and HR professionals. When an HR professional feeds the platform a job description and a batch of resumes (including PDFs, Word documents, or images), HRScreener automatically extracts essential candidate information, evaluates semantic fit, and produces a scored shortlist. Rather than relying on simple keyword matching, it leverages advanced Vision-Language Models (VLMs) and Generative AI to truly "read" resumes like a human would—mapping out skill hierarchies, calculating career progression, and identifying implicit strengths or crucial missing skills. The result is a highly readable hiring recommendation and scoring report that guides the interview process and empowers hiring managers to make confident, data-driven decisions.

---

## 1. PROJECT OVERVIEW

**What the project does:**
HRScreener parses unstructured candidate resumes and compares them against a predefined Job Description (JD). It ranks candidate profiles, points out critical missing skills, and provides focused feedback for interviewers.

**The core problem it solves:**
Recruiters spend a significant portion of their workload scanning unformatted, unstructured resumes. Traditional ATS (Applicant Tracking Systems) use naive keyword matching (which misses semantic overlaps like "React" vs "Next.js" or "GCP" vs "AWS"). HRScreener utilizes AI to bridge this gap, ensuring qualified candidates aren't dropped due to formatting quirks and providing substantive reasoning for each rejection or progression.

**Who the end users are:**
- **HR Professionals / Recruiters:** looking for faster initial screening.
- **Hiring Managers:** looking for transparent technical summaries of shortlisted candidates.

**Tech Stack Summary:**
- **Backend Framework:** Python, FastAPI, Uvicorn
- **AI & NLP:** PyTorch, Transformers, Qwen2-VL (Advanced Vision-Language Model), Groq API / Google Gemini API
- **Document Processing:** PyTesseract (OCR), Poppler (pdf2image), python-docx, pypdf
- **Data Validation:** Pydantic
- **Frontend Framework:** React, Vite (Javascript)

---

## 2. ARCHITECTURE OVERVIEW

**High-Level System Design:**
The core backend is structured around a modular FastAPI pipeline. The architecture is split between Document Perception (OCR + Layout), Data Extraction (Vision Models + Fallbacks), Intelligence/Enrichment (LLM via Gemini/Groq), and Scoring (Heuristic/Semantic pipelines).

**Data Flow (Step by Step):**
1. **Input Submission:** The React frontend POSTs a job description and a list of resume files (PDF, DOCX, Img) to the `/screen-resumes` API endpoint.
2. **Document Parsing:** The `DocumentParser` standardizes inputs. PDFs are converted to a text stream via `pypdf`, and rasterized into images via Poppler.
3. **Extraction (`HRExtractionService`):**
   - Resumes and JDs are passed to the Qwen2-VL model. Because it is a Vision-Language Model, it 'sees' the resume layout alongside the Tesseract-extracted OCR text, highly minimizing parsing errors common in complex multi-column resumes.
   - Pydantic models validate the JSON output. If the VLM hallucinates or truncates JSON, custom robust healing logic attempts to fix it.
   - If the local VLM fails, the system falls back to the Gemini LLM service using OCR text.
4. **Data Enrichment:** The `LLMUnderstandingService` dynamically extracts implicit candidate personas (e.g., "backend-heavy developer") and groups technological skills into semantic trees.
5. **Evaluation & Scoring:** The structured data passes into `scoring.py`. Semantic matchers calculate proximity between skills, compute a candidate's career trajectory (from experience dates), and generate a percentage fit.
6. **Reasoning:** Finally, a generative model or heuristic pipeline produces readable rationale (Interview Focus Areas, Secret Strengths) indicating exactly *why* a candidate was scored this way.
7. **Response:** A ranked JSON response containing the synthesized candidate scores and recruiter feedback is surfaced back to the user.

**External Services and APIs:**
- **Google Gemini API / Groq:** Used as an intelligent enrichment layer. They create "intelligence" over the baseline extracted facts (inferring seniority, clustering skills) with much higher reasoning capabilities than regex.
- **Tesseract & Poppler binaries:** Run locally to brute-force document layout reading and rasterization.

---

## 3. COMPONENT BREAKDOWN

### `main.py`
- **What it does:** The primary FastAPI router and application lifecycle hook.
- **Key Functions:** `screen_resumes()` endpoint handles the multipart/form-data. It integrates an `asyncio.Semaphore` limit to execute a maximum of 4 files concurrently, protecting OS memory. 
- **Design Decisions:** Used `asyncio.to_thread` for CPU-bound extraction work so the event loop doesn’t block, maintaining a highly responsive API scale.

### `app/agent.py`
- **What it does:** Orchestrates the `HRExtractionService`. Houses the prompt engineering and the Vision-Language Model pipeline.
- **Key Functions:** `extract_resume()`, `generate_response()`, `parse_json()`.
- **Design Decisions:** Utilizes custom `_repair_truncated_json()` to rescue syntactically truncated outputs from token-limited LLMs instead of aggressively failing. Manually invokes `torch.cuda.empty_cache()` inside an `_inference_lock` to ensure multiple concurrent requests don't cause GPU OOM (Out Of Memory) states.

### `app/document_parser.py` & `app/ocr.py`
- **What they do:** Handles the translation of binary blobs into standardized, cleaned text and PIL images. They orchestrate Poppler/PyPDF/Tesseract.
- **Key Functions:** `extract_with_boxes()`, `annotate_resume_sections()`.
- **Design Decisions:** Instead of blindly passing OCR text to the LLM, `ocr.py` utilizes Tesseract's bounding box data (`image_to_data`) to identify column breaks and append distinct markers like `[CONTACT_INFO]` or `[SKILLS]`. This vastly reduces VLM extraction hallucinations.

### `app/schemas.py`
- **What it does:** Pydantic V2 classes defining the data boundaries of the system.
- **Key Features:** Uses strong `@field_validator` hooks to perform complex business logic directly on instantiation. For example, `compute_years_of_experience()` mathematical calculates tenure based on date gaps, while `filter_education_from_experience()` prevents the LLM from passing "Student/Intern" data into the Work Experience block.

### `app/scoring.py` & `app/semantic_matching.py`
- **What they do:** Evaluates candidate fits by crunching algorithmic overlaps rather than exact string matches.
- **Key Algorithms:** Calculates career progression velocity, domain overlap via contextual keyword counting (`domain_bonus_score`), and uses `SemanticMatcher` so that related terms (e.g. `React.js` and `JavaScript`) score accurately without manual mapping tables.

### `app/reasoning.py` & `app/llm_understanding.py`
- **What they do:** Abstract the LLM interactions for "understanding" rather than purely "extracting". 
- **Design Decisions:** `LLMUnderstandingService` implements localized caching (`skill_graph_cache.json`) to prevent calling Groq/Gemini APIs redundantly for commonly encountered skills (e.g. "React"), dramatically saving latency and cost.

---

## 4. KEY TECHNICAL DECISIONS

1. **VLM (Vision-Language Model) Over Standard LLM:**
   By passing actual rendered images (via Qwen2-VL) rather than just raw text to the extraction model, the system avoids entirely misinterpreting modern multi-column resumes. An LLM sees a garbled mush of text. A VLM visually understands where the "Experience" column ends and the "Projects" column begins.
2. **Pydantic Driven Business Logic:**
   Moving the data sanitation (like date calculations, Regex email cleaning, string normalizing) directly into Pydantic validators ensures that any downstream function receives guaranteed, sanitized data. It keeps models extremely robust.
3. **Locking GPU Inference:**
   While the API processes inputs asynchronously using `asyncio`, actual PyTorch inference is throttled via a `threading.Lock()` block and semaphore limits. This trades minor speed reduction for total system stability (avoiding hard VRAM crashes under load).
4. **Resilient JSON Healing:**
   Generative models occasionally produce trailing commas or truncate ending braces. Rather than failing the request and incurring a massive retry latency, the system utilizes string manipulation (`_repair_truncated_json`) to forcefully close open brackets and parse whatever data *was* retrieved.

---

## 5. CHALLENGES FACED & HOW THEY WERE SOLVED

- **Challenge:** LLMs confusing Education coursework with Professional Experience.
  - **Solution:** While Prompt engineering improved this, the definitive fix was implementing a heuristic filter inside `schemas.py` (`filter_education_from_experience`), which cross-references title tokens against academic keywords (B.Sc, PhD, Student).
- **Challenge:** LLM Outputs Exceeding Context/Token Limits resulting in broken JSON.
  - **Solution:** Built a 3-tier safety net. Tier 1: Instruct model to be concise. Tier 2: `_repair_truncated_json()`. Tier 3: If all else fails, use a secondary cheaper LLM API fallback layer (Gemini).
- **Challenge:** Variations in Skill Terminology (e.g., "Node", "NodeJS", "Node.js").
  - **Solution:** Built `semantic_matching.py` alongside an LLM-powered cached `skill_graph` to dynamically parent distinct keywords into identical conceptual boundaries. 
- **Challenge:** Memory Exhaustion during PDF parsing.
  - **Solution:** Enforced strict boundaries on `MAX_PAGES_PER_RESUME` (cutting off large, multi-page PDFs at 5 pages) through config validation.

---

## 6. FEATURES & CAPABILITIES

### Implemented Features
- **Visual Resume parsing:** Handles heavily formatted PDFs via multimodal capabilities.
- **Implicit Intelligence:** Reads JD responsibilities and deduces implicit skills (What technical skills are explicitly *not* mentioned but completely mandatory for this job).
- **Risk Assessment:** Flags critical missing skills and suggests explicit Focus Areas for human interviewers to probe.
- **Data Normalization:** Automatically standardizes dates, determines precise experience tenure in months, and ranks seniority level.
- **Recruiter Feedback Generation:** Summarizes the entire algorithmic score into one natural, human-readable paragraph.

### Known Limitations & Edge Cases
- **Non-Textual Data:** Graphical charts describing skill competency (e.g. 5 stars in Photoshop) are difficult for the VLM to transpose accurately to metrics.
- **Local Compute Costs:** Processing through a local 2B+ parameter multimodal model can be slow on machines lacking dedicated compute nodes or NVidia GPUs.

---

## 7. SETUP & DEPLOYMENT

### External Infrastructure & Binaries
To run this application, the host machine **must** have:
- `Tesseract OCR` installed.
- `Poppler` utils installed (to support rasterizing PDFs to images).
- `CUDA` / PyTorch backend available (for optimal inference performance).

### Environment Variables Required
Create a `.env` file referencing the following:
```env
MODEL_ID="Qwen/Qwen2-VL-2B-Instruct" 
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe" # Or /usr/bin/tesseract
POPPLER_PATH="D:\Work\Poppler\poppler-25.12.0\Library\bin" # Required on Windows
MAX_RESUMES_PER_REQUEST=10
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="your-groq-key"
```

### Running Locally
1. Navigate to the backend directory `cd backend`.
2. Install Python dependencies: `pip install -r requirements.txt`.
3. Boot the FastAPI server: `uvicorn app.main:app --reload` or use `fastapi dev app/main.py`.
4. In a separate terminal, run the frontend via `cd Frontend && npm install && npm run dev`.

### Deployment (Target Strategy)
Best deployed using Docker containers. The `Dockerfile` should specify a base Python image and natively install `poppler-utils` and `tesseract-ocr` via `apt-get` prior to `pip install` to ensure cross-compatibility in Linux server environments.

---

## 8. WHAT I LEARNED FROM THIS PROJECT

- **The Value of Multimodal Output:** Encountering traditional OCR boundaries natively highlighted why OCR + NLP is falling behind. VLMs mapping textual features to specific layout placements is drastically more effective.
- **Robustness in Generative Systems:** Creating a GenAI application is only 20% prompting, and 80% fallbacks, typing, structural checks, and retry mechanisms.
- **Semantic Over Literal:** Hard-coding skill definitions is futile. By relying heavily on intelligent vector proximity maps (`SemanticMatcher`) and LLM skill-trees, systems adapt to new tech stacks natively without maintenance overhead.

---

## 9. FUTURE IMPROVEMENTS

- **Persistent Vector Summarization:** If we integrate an external Vector Database (like Pinecone, Milvus, or Postgres using `pgvector`), we can retain candidates from past jobs. This would enable a feature like *"Query past candidates matching this new JD."*
- **Async Queueing Architecture:** Move `score_candidate` and extraction logic off the FastAPI request thread entirely and into a Background Queue system like `Celery` + `Redis/RabbitMQ` so end users can close the browser and receive an email when their batch is done.
- **Interactive Chat Assistant:** Give hiring managers a chat interface on the frontend to query: "Why did Candidate X lack progression in backend infra compared to Candidate Y?"
