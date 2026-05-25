import re
from typing import Any


def infer_role_weight(title: str | None, company: str | None) -> float:
    title = title.lower() if title else ""
    company = company.lower() if company else ""

    # Internship
    if "intern" in title:
        return 0.5
    
    # Freelance
    if any(word in title for word in ["freelance", "freelancer", "self-employed"]):
        return 0.7

    # Contract roles
    if "contract" in title:
        return 0.8
    
    # Default -> Fulltime
    return 1.0


def filter_education_from_experience_entries(entries: list[Any]) -> list[Any]:
    filtered = []
    degree_kw = {"btech", "mtech", "bachelor", "bachelors", "master", "masters", "bsc", "msc", "phd", "degree", "diploma"}
    edu_inst_kw = {"institute", "university", "college", "school", "academy"}
    work_kw = {
        "assistant", "researcher", "professor", "lecturer", "intern", 
        "developer", "dev", "engineer", "software", "fullstack", "frontend", "backend",
        "manager", "lead", "coordinator", "tutor", "instructor", "faculty", 
        "staff", "postdoc", "fellow", "consultant", "freelance", "freelancer", 
        "founder", "mentor", "creator", "writer", "designer", "editor", "analyst",
        "specialist", "executive", "administrator", "scientist", "architect", 
        "technician", "associate", "expert", "officer", "director", "head", 
        "principal", "president", "vp", "ui", "ux"
    }

    for entry in entries:
        title_lower = entry.title.lower()
        company_lower = entry.company.lower()
        
        title_words = set(re.findall(r'\b[a-z]+\b', title_lower))
       
        # 1. If it explicitly spells out b.tech or m.tech with dots
        if any(term in title_lower for term in ["b.tech", "m.tech", "b.e.", "b.sc", "m.sc"]):
            continue
            
        # 2. If title words contain degree keywords
        if degree_kw & title_words:
            continue
        
        job_match_strength = len(work_kw & title_words)
        edu_match_strength = len(degree_kw & title_words)

        if "student" in title_words or "undergraduate" in title_words:
            edu_match_strength += 2 

        is_job = job_match_strength >=1 and job_match_strength > edu_match_strength 

        if not is_job:
            continue 

        if entry.duration_months >= 36:
            if "student" in title_lower or any(
                word in company_lower for word in ["college", "university", "institute"]
                ):
                    continue

        if any(word in company_lower for word in ["college", "university", "institute", "school"]):
            if "intern" not in title_lower:
                continue
        
        filtered.append(entry)          
    return filtered


def compute_years_of_experience(entries: list[Any], current_total: float) -> float:
    weighted_months = 0.0

    for entry in entries:
        weight = infer_role_weight(entry.title, entry.company)
        weighted_months += entry.duration_months * weight

    calculated_years = round(weighted_months / 12, 1)

    if not entries:
        return 0.0
    elif calculated_years > 0:
        if calculated_years >= current_total or current_total == 0.0:
            return calculated_years
    
    return current_total
