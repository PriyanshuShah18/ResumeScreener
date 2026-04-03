import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas import JobDescriptionData, ResumeData

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

SKILL_GRAPH_CACHE_FILE = Path("skill_graph_cache.json")

class LLMUnderstandingService:
    def __init__(self):
        self.settings = get_settings()
        self.enabled = False
        self.provider = None

        if self.settings.groq_api_key in str(self.settings.groq_api_key):
            # Prioritize Groq if provided
            key = self.settings.groq_api_key or "gsk_"
            if Groq:
                self.client = Groq(api_key=key)
                self.provider = "groq"
                self.enabled = True
            else:
                logger.warning("Groq key found, but groq library not installed.")
        elif self.settings.gemini_api_key and genai:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            self.provider = "gemini"
            self.enabled = True
        else:
            logger.warning("No LLM API keys found. LLM Understanding disabled.")

    def _generate_json(self, prompt: str, default: Any) -> Any:
        if not self.enabled:
            return default
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            elif self.provider == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(response_mime_type="application/json")
                )
                return json.loads(response.text)
        except Exception as exc:
            logger.error("LLM Generation failed: %s", exc)
            return default

    def enrich_jd_intelligence(self, job: JobDescriptionData) -> JobDescriptionData:
        prompt = f"""
        Analyze this Job Description and extract implicit intelligence.
        Title: {job.title}
        Responsibilities: {job.responsibilities}
        Must Have Skills: {job.must_have_skills}
        
        Return a JSON object:
        {{
            "implicit_skills": ["list of implicit skills (e.g. Distributed Systems)"],
            "inferred_seniority": "Senior/Mid/Junior",
            "domain_expectations": ["Industry domain expectations"]
        }}
        """
        result = self._generate_json(prompt, {})
        job.implicit_skills = result.get("implicit_skills", [])
        job.inferred_seniority = result.get("inferred_seniority", "")
        job.domain_expectations = result.get("domain_expectations", [])
        return job

    def generate_skill_graph(self, skills: list[str]) -> dict[str, list[str]]:
        if not skills:
            return {}
            
        cache = self._load_cache()
        uncached = [skill for skill in skills if skill not in cache]
        
        if not uncached:
            return {s: cache[s] for s in skills if s in cache}
            
        prompt = f"""
        Group these skills into broader hierarchies/categories representing related domains.
        Skills: {uncached}
        
        Return a JSON object where the KEY is the original skill, and the VALUE is a list of broader parent categories.
        Example: {{"react": ["frontend", "javascript", "ui engineering"]}}
        """
        result = self._generate_json(prompt, {})
        
        if not isinstance(result, dict):
            result = {}
            
        for skill in uncached:
            if skill not in result or not isinstance(result[skill], list):
                result[skill] = [skill]
                
        cache.update(result)
        self._save_cache(cache)
        
        return {s: cache[s] for s in skills if s in cache}

    def _load_cache(self) -> dict[str, list[str]]:
        if SKILL_GRAPH_CACHE_FILE.exists():
            try:
                with open(SKILL_GRAPH_CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict[str, list[str]]):
        try:
            with open(SKILL_GRAPH_CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save skill graph cache: %s", exc)

    def enrich_candidate_persona(self, resume: ResumeData, job: JobDescriptionData) -> ResumeData:
        prompt = f"""
        Review the candidate's core skills and compare them to the Job Requirements to create a recruiting persona.
        Candidate Skills: {resume.skills + resume.tools}
        Job Required Skills: {job.must_have_skills}
        
        Return a JSON object answering the schema:
        {{
            "enriched_persona": [
                "Backend-heavy engineer",
                "Has PyTorch, can easily adapt to TensorFlow",
                "Lacks explicit Kubernetes, but strong AWS/Docker implies easy ramp up"
            ]
        }}
        """
        result = self._generate_json(prompt, {})
        resume.enriched_persona = result.get("enriched_persona", [])
        resume.skill_clusters = self.generate_skill_graph(resume.skills + resume.tools)
        return resume

    def extract_json_from_text(self, text: str, schema_json: dict) -> dict | None:
        """Fallback JSON extraction from raw OCR text using Gemini."""
        if not self.enabled or not text.strip():
            return None
        prompt = f"""
        Extract the following text into the provided JSON schema. Ensure valid JSON output only.
        
        Text:
        {text[:15000]}
        
        Schema:
        {json.dumps(schema_json)}
        """
        return self._generate_json(prompt, None)

llm_service = LLMUnderstandingService()
