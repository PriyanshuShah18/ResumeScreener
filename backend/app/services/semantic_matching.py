from __future__ import annotations

import math
import os
import re
import threading
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

import httpx
from app.core.constants import SKILL_ALIASES
from app.utils.text_utils import normalize_skill

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False



def canonicalize_skill(value: str) -> str:
    normalized = normalize_skill(value)
    return SKILL_ALIASES.get(normalized, normalized)


def _tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\+\#\./]+", canonicalize_skill(value)))


def _is_acronym_match(left: str, right: str) -> bool:
    left_tokens = [token for token in canonicalize_skill(left).replace(".", " ").split() if token]
    right_tokens = [token for token in canonicalize_skill(right).replace(".", " ").split() if token]
    if not left_tokens or not right_tokens:
        return False

    left_acronym = "".join(token[0] for token in left_tokens)
    right_acronym = "".join(token[0] for token in right_tokens)
    left_single = len(left_tokens) == 1 and len(left_tokens[0]) <= 5
    right_single = len(right_tokens) == 1 and len(right_tokens[0]) <= 5
    return (left_single and left_tokens[0] == right_acronym) or (right_single and right_tokens[0] == left_acronym)


def lexical_similarity(left: str, right: str) -> float:
    left_norm = canonicalize_skill(left)
    right_norm = canonicalize_skill(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if _is_acronym_match(left_norm, right_norm):
        return 0.92

    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        has_subset = left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)
        if has_subset and min(len(left_tokens), len(right_tokens)) >= 2:
            subset_bonus = 0.9
        elif has_subset:
            subset_bonus = 0.85
        else:
            subset_bonus = 0.0
    else:
        jaccard = 0.0
        subset_bonus = 0.0

    sequence_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence_ratio, subset_bonus)


class SemanticMatcher:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.75,
        partial_threshold: float = 0.60,
        required_match_threshold: float = 0.88,
        additional_relevance_threshold: float = 0.60,
        clustering_threshold: float = 0.78,
        domain_bonus_max: int = 8,
        enable_model: bool = True,
    ) -> None:
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.partial_threshold = partial_threshold
        self.required_match_threshold = required_match_threshold
        self.additional_relevance_threshold = additional_relevance_threshold
        self.clustering_threshold = clustering_threshold
        self.domain_bonus_max = max(domain_bonus_max, 0)
        
        self._model = None
        self._model_lock = threading.Lock()
        # Embedding cache stores text -> numpy array (from HF) or pytorch tensor (from local model)
        self._embedding_cache: dict[str, Any] = {}
        self._cache_max = 2048
        self.enable_model = enable_model and _HAS_TRANSFORMERS
        self._hf_api_key: str | None = None
        self._hf_model_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        # Circuit breaker: flipped to False on first HF failure.
        # Protected by a lock so parallel resume threads don't all race past the check simultaneously.
        self._hf_healthy: bool = True
        self._hf_lock = threading.Lock()

    @property
    def model(self):
        if not self.enable_model:
            return None
        with self._model_lock:
            if self._model is None:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as e:
                    print(f"Failed to load semantic model {self.model_name}: {e}")
                    self.enable_model = False
            return self._model

    def _hf_embed_batch(self, texts: list[str]) -> list[Any] | None:
        """Fetch embeddings for a list of texts from HuggingFace Serverless API in ONE request.

        This is the bulk pre-computation method. Results are returned as a list of
        raw embedding vectors (plain Python lists of floats) in the same order as `texts`.
        Returns None if the API is unavailable, the key is not set, or the circuit is open.
        The _hf_lock ensures only one thread trips the breaker and prints the log message.
        """
        if not self._hf_api_key or not texts:
            return None
        # Fast path: check without lock (already tripped)
        if not self._hf_healthy:
            return None
        with self._hf_lock:
            # Re-check inside lock in case another thread tripped it while we waited
            if not self._hf_healthy:
                return None
            try:
                response = httpx.post(
                    self._hf_model_url,
                    headers={"Authorization": f"Bearer {self._hf_api_key}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=20.0,
                )
                if response.status_code != 200:
                    print(f"HuggingFace API error {response.status_code}: {response.text[:200]}")
                    self._hf_healthy = False
                    return None
                data = response.json()
                if isinstance(data, list) and len(data) == len(texts):
                    return data
                return None
            except Exception as exc:
                # Trip the circuit breaker: log ONCE, then go silent for the rest of the session
                print(f"HuggingFace API unreachable ({exc}). Falling back to local model for this session.")
                self._hf_healthy = False
                # Re-enable local model as a fallback if it was disabled when HF key was configured
                if not self.enable_model and _HAS_TRANSFORMERS:
                    self.enable_model = True
                return None

    def precompute_hf_embeddings(self, terms: list[str]) -> None:
        """Pre-warm the embedding cache for all given terms in a single HF API call.

        Call this ONCE before a batch of similarity() calls (e.g. at the start of
        match_skills). Subsequent similarity() calls will read from the in-memory
        cache and never hit the API again — making per-pair comparison instant.
        """
        if not self._hf_api_key or not terms:
            return
        import numpy as np
        # Only fetch terms that are not already cached
        uncached = [t for t in terms if t not in self._embedding_cache]
        if not uncached:
            return
        embeddings = self._hf_embed_batch(uncached)
        if embeddings is None:
            return
        for text, emb_list in zip(uncached, embeddings):
            emb = np.array(emb_list, dtype="float32")
            norm = np.linalg.norm(emb)
            if norm > 0:
                # Store the L2-normalised vector so cosine similarity = just a dot product
                self._embedding_cache[text] = emb / norm
            else:
                self._embedding_cache[text] = emb


    def _normalize_terms(self, values: list[str], canonicalize: bool = True) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            term = canonicalize_skill(value) if canonicalize else normalize_skill(value)
            if not term or term in seen:
                continue
            seen.add(term)
            normalized.append(term)
        return normalized

    def similarity_credit(self, similarity: float) -> float:
        if similarity >= self.semantic_threshold:
            return 1.0
        if similarity >= self.partial_threshold:
            return 0.5
        return 0.0

    def best_match(self, query: str, candidates: list[str]) -> tuple[str, float]:
        if not query or not candidates:
            return "", 0.0

        best_candidate = ""
        best_score = 0.0
        for candidate in candidates:
            current_score = self.similarity(query, candidate)
            if current_score > best_score:
                best_candidate = candidate
                best_score = current_score
        return best_candidate, best_score

    def match_skills(self, jd_skills: list[str], resume_skills: list[str], threshold: float | None = None) -> dict[str, Any]:
        required_threshold = self.required_match_threshold if threshold is None else threshold
        normalized_jd = self._normalize_terms(jd_skills)
        normalized_resume = self._normalize_terms(resume_skills)

        # Dynamically populate the skill graph cache for any unseen skills
        all_skills = list(set(normalized_jd + normalized_resume))
        skill_graph_default = _get_bool_env("SEMANTIC_MODEL_ENABLED", True)
        if all_skills and _get_bool_env("SEMANTIC_SKILL_GRAPH_ENABLED", skill_graph_default):
            from app.services.llm_understanding import llm_service
            try:
                llm_service.generate_skill_graph(all_skills)
            except Exception as e:
                print(f"Failed to populate skill graph: {e}")

        # Pre-compute ALL embeddings in ONE batched HF API call before looping.
        # This avoids making an API call per similarity() invocation (which would be
        # O(jd_skills × resume_skills) requests — the bug that caused 598s runtimes).
        if self._hf_api_key:
            self.precompute_hf_embeddings(all_skills)

        matched: list[str] = []
        missing: list[str] = []
        details: list[dict[str, Any]] = []
        matched_candidates: set[str] = set()

        for jd_skill in normalized_jd:
            matched_skill, similarity = self.best_match(jd_skill, normalized_resume)
            is_match = bool(matched_skill) and similarity >= required_threshold
            if is_match:
                matched.append(jd_skill)
                matched_candidates.add(matched_skill)
            else:
                missing.append(jd_skill)

            details.append(
                {
                    "jd_skill": jd_skill,
                    "matched_skill": matched_skill,
                    "similarity": round(similarity, 3),
                    "is_match": is_match,
                    "threshold": round(required_threshold, 3),
                }
            )

        return {
            "matched": matched,
            "missing": missing,
            "details": details,
            "matched_candidates": sorted(matched_candidates),
        }

    def find_additional_relevant_skills(
        self,
        jd_context_skills: list[str],
        resume_skills: list[str],
        exclude: set[str] | None = None,
        threshold: float | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        relevance_threshold = self.additional_relevance_threshold if threshold is None else threshold
        normalized_jd = self._normalize_terms(jd_context_skills)
        normalized_resume = self._normalize_terms(resume_skills)
        excluded = {canonicalize_skill(item) for item in (exclude or set()) if canonicalize_skill(item)}

        ranked: list[tuple[str, str, float]] = []
        for resume_skill in normalized_resume:
            if resume_skill in excluded:
                continue
            jd_match, similarity = self.best_match(resume_skill, normalized_jd)
            if jd_match and similarity >= relevance_threshold:
                ranked.append((resume_skill, jd_match, similarity))

        ranked.sort(key=lambda item: item[2], reverse=True)

        relevant_skills: list[str] = []
        details: list[dict[str, Any]] = []
        seen: set[str] = set()
        for resume_skill, jd_match, similarity in ranked:
            if resume_skill in seen:
                continue
            seen.add(resume_skill)
            relevant_skills.append(resume_skill)
            details.append(
                {
                    "resume_skill": resume_skill,
                    "jd_context_match": jd_match,
                    "similarity": round(similarity, 3),
                    "threshold": round(relevance_threshold, 3),
                }
            )

        return relevant_skills, details

    def cluster_skills(self, skills: list[str], threshold: float | None = None) -> list[dict[str, Any]]:
        cluster_threshold = self.clustering_threshold if threshold is None else threshold
        normalized_skills = self._normalize_terms(skills, canonicalize=False)
        clusters: list[dict[str, Any]] = []

        for skill in normalized_skills:
            best_cluster_index = -1
            best_similarity = 0.0

            for index, cluster in enumerate(clusters):
                cluster_center = cluster["canonical"]
                similarity = self.similarity(skill, cluster_center)
                if similarity >= cluster_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_index = index

            if best_cluster_index >= 0:
                cluster = clusters[best_cluster_index]
                members: list[str] = cluster["members"]
                if skill not in members:
                    members.append(skill)
                continue

            clusters.append({"canonical": skill, "members": [skill]})

        for cluster in clusters:
            cluster["members"] = sorted(cluster["members"])

        clusters.sort(key=lambda item: item["canonical"])
        return clusters

    def detect_domain_tags(
        self,
        jd_terms: list[str],
        resume_terms: list[str],
        threshold: float | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        domain_threshold = self.additional_relevance_threshold if threshold is None else threshold
        jd_clusters = self.cluster_skills(jd_terms)
        resume_clusters = self.cluster_skills(resume_terms)

        resume_centers = [cluster["canonical"] for cluster in resume_clusters]
        domain_hits: list[tuple[str, str, float]] = []

        for jd_cluster in jd_clusters:
            jd_center = jd_cluster["canonical"]
            matched_resume, similarity = self.best_match(jd_center, resume_centers)
            if matched_resume and similarity >= domain_threshold:
                domain_hits.append((jd_center, matched_resume, similarity))

        domain_hits.sort(key=lambda item: item[2], reverse=True)

        tags: list[str] = []
        details: list[dict[str, Any]] = []
        seen: set[str] = set()
        for jd_tag, matched_resume, similarity in domain_hits:
            if jd_tag in seen:
                continue
            seen.add(jd_tag)
            tags.append(jd_tag)
            details.append(
                {
                    "jd_domain_tag": jd_tag,
                    "matched_resume_term": matched_resume,
                    "similarity": round(similarity, 3),
                    "threshold": round(domain_threshold, 3),
                }
            )

        return tags, details

    def similarity(self, left: str, right: str) -> float:
        left_norm = canonicalize_skill(left)
        right_norm = canonicalize_skill(right)
        if not left_norm or not right_norm:
            return 0.0
        if left_norm == right_norm:
            return 1.0

        # Fallback 1: Lexical exact-string overlaps (instant)
        lexical_score = lexical_similarity(left_norm, right_norm)
        if lexical_score >= 0.95:
            return lexical_score

        # Primary Intelligence: Check Gemini LLM dynamic skill graph
        from app.services.llm_understanding import llm_service, skill_graph_terms
        try:
            llm_cache = llm_service._load_cache()
            left_terms = [s.lower() for s in skill_graph_terms(llm_cache.get(left_norm))]
            right_terms = [s.lower() for s in skill_graph_terms(llm_cache.get(right_norm))]
            if left_norm in llm_cache and right_norm in left_terms:
                return 0.85
            if right_norm in llm_cache and left_norm in right_terms:
                return 0.85
        except Exception:
            pass

        # Fallback 2a: HuggingFace cached embeddings (pre-computed — instant numpy dot product)
        # precompute_hf_embeddings() is called once in match_skills() before any loop.
        # If HF was unavailable, the cache will be empty and we fall through to the local model.
        import numpy as np
        left_emb = self._embedding_cache.get(left_norm)
        right_emb = self._embedding_cache.get(right_norm)

        if left_emb is not None and right_emb is not None:
            try:
                # Embeddings stored pre-normalised, so cosine sim = dot product
                hf_score = float(np.dot(left_emb, right_emb))
                return max(lexical_score, hf_score)
            except Exception:
                pass

        # Fallback 2b: Local Sentence-BERT Embeddings (offline fallback, with cache)
        if self.enable_model and self.model:
            try:
                def _get_cached_embedding(text):
                    if text in self._embedding_cache:
                        return self._embedding_cache[text]
                    emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                    if len(self._embedding_cache) < self._cache_max:
                        self._embedding_cache[text] = emb
                    return emb

                emb_left = _get_cached_embedding(left_norm)
                emb_right = _get_cached_embedding(right_norm)
                from sentence_transformers import util
                semantic_score = float(util.cos_sim(emb_left, emb_right)[0][0])
                return max(lexical_score, semantic_score)
            except Exception:
                pass

        return lexical_score


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float_env(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _get_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


@lru_cache(maxsize=1)
def get_semantic_matcher() -> SemanticMatcher:
    from app.core.config import get_settings
    settings = get_settings()
    hf_key = settings.hf_api_key or os.getenv("HF_API_KEY")
    # When HF API is active, disable the local PyTorch model — it's not needed
    # and loading it wastes RAM and startup time on cloud servers.
    local_model_default = not bool(hf_key)
    matcher = SemanticMatcher(
        model_name=os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2"),
        semantic_threshold=_get_float_env("SEMANTIC_THRESHOLD", 0.75),
        partial_threshold=_get_float_env("SEMANTIC_PARTIAL_THRESHOLD", 0.60),
        required_match_threshold=_get_float_env("SEMANTIC_REQUIRED_THRESHOLD", 0.88),
        additional_relevance_threshold=_get_float_env("SEMANTIC_ADDITIONAL_THRESHOLD", 0.60),
        clustering_threshold=_get_float_env("SEMANTIC_CLUSTER_THRESHOLD", 0.78),
        domain_bonus_max=_get_int_env("SEMANTIC_DOMAIN_BONUS_MAX", 8),
        enable_model=_get_bool_env("SEMANTIC_MODEL_ENABLED", local_model_default),
    )
    if hf_key:
        matcher._hf_api_key = hf_key
        print(f"✅ HuggingFace Serverless Inference API enabled — local model disabled")
    return matcher
