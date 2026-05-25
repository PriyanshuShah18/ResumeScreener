# HR-Screening

**An Intelligent, Multimodal Resume Screening System**

HR-Screening is an automated recruitment platform built to alleviate the manual burden placed on HR professionals. By feeding the platform a job description and a batch of candidate resumes (PDFs, Word docs, or images), HR-Screening automatically extracts essential candidate information, evaluates semantic fit, and produces a scored shortlist.

Unlike traditional Applicant Tracking Systems (ATS) that rely on naive, rigid keyword matching, HR-Screening leverages **Vision-Language Models (VLMs)** and Generative AI to truly "read" resumes like a human would. It visually understands layout hierarchies, calculates career progression velocity, semantically groups overlapping skills, and identifies crucial missing capabilities. 

The result is a highly readable hiring recommendation and scoring report that provides substantive reasoning for each candidate, empowering hiring managers to make confident, data-driven decisions.

---

## Key Features

- **Multimodal Visual Parsing:** Utilizes Qwen2-VL to "see" the resume layout alongside OCR text, drastically reducing parsing errors common in complex, multi-column resumes.
- **Implicit Intelligence:** Reads job description responsibilities and deduces implicit skills (e.g. recognizing that knowing "React" implies frontend capabilities).
- **Semantic Over Literal Matching:** Uses semantic vector matching to identify that terms like "Node" and "Node.js" are conceptually identical without requiring manual mapping tables.
- **Data Normalization:** Automatically standardizes dates, determines precise experience tenure in months, and ranks seniority levels.
- **Risk Assessment & Feedback:** Flags critical missing skills and suggests explicit Interview Focus Areas for human interviewers to probe. Generates a natural, human-readable paragraph explaining the candidate's algorithmic score.
- **Robust Fallbacks:** Implements custom Pydantic validators, JSON healing logic, and dynamic LLM fallback tiers (Google Gemini / Groq) to ensure the system is resilient against token limits and generative hallucinations.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **AI & NLP:** PyTorch, Transformers, Qwen2-VL, Google Gemini API, Groq API
- **Document Processing:** PyTesseract (OCR), Poppler (pdf2image), pypdf, python-docx
- **Data Validation:** Pydantic (V2)
- **Database:** MongoDB
- **Frontend:** React, Vite, Javascript

---

## Local Development & Setup

### Prerequisites
To run this application locally, your machine **must** have:
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to PATH.
- [Poppler](https://poppler.freedesktop.org/) installed (for rasterizing PDFs to images).

### Environment Variables
Create a `.env` file in the `backend/` directory referencing the following (see `.env.example` for reference):
```env
MODEL_ID="Qwen/Qwen2-VL-2B-Instruct" 
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe" # Or /usr/bin/tesseract
POPPLER_PATH="D:\Work\Poppler\poppler-25.12.0\Library\bin" # Required on Windows
MAX_RESUMES_PER_REQUEST=10
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="your-groq-key"
```

### Start Commands

You should run each of these services in a separate terminal window from the root of the project (`e:\HR-Screening`).

**1. Run the Backend API (FastAPI)**
```bash
cd backend
uvicorn app.main:app --reload
# Alternatively: fastapi dev app/main.py
```

**2. Run the ATS Background Worker**
This worker polls MongoDB to process pending job applications asynchronously.
```bash
cd backend
python -m app.workers.ats_worker
# Options: --once, --ensure-indexes, --clear-parsed-cache
```

**3. Run the Frontend (React/Vite)**
```bash
cd Frontend
npm install
npm run dev
```

---

## Project Structure

- **`backend/app/`**: The FastAPI service. It follows a domain-driven technical pattern:
  - `core/`: Application settings and constants.
  - `schemas/`: Pydantic V2 data boundaries and robust input validators.
  - `services/`: Business logic, LLM orchestrations (`agent.py`), OCR processing, semantic matchers, and scoring algorithms.
  - `utils/`: NLP and text sanitization helpers.
  - `workers/`: Background polling scripts (`ats_worker.py`).
- **`Frontend/`**: React + Vite application for the user interface.
- **`docs/`**: Comprehensive project documentation.

---

## Documentation Deep-Dive

For a complete breakdown of the architectural design, API flows, challenges solved, and future improvement plans, please review the documentation artifacts:
- [Project Documentation](docs/PROJECT_DOCUMENTATION.md)
- [Change Log](docs/PROJECT_CHANGE_LOG.md)
