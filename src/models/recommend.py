from typing import List, Dict
import pandas as pd
import os

# Example mapping: symptom/tag to module
SYMPTOM_TO_MODULE = {
    "fever": "Malaria Prevention",
    "cough": "ARI Recognition",
    "pregnancy": "Maternal Health",
    "diarrhea": "Diarrhea Management",
    "chills": "Malaria Prevention",
    "dehydration": "Diarrhea Management",
    "breathing": "ARI Recognition",
}

MODULES = ["Malaria Prevention", "Diarrhea Management", "ARI Recognition", "Maternal Health"]


def recommend_modules(region: str, patient_tags: List[str], level: str) -> Dict:
    """
    Recommends learning modules based on region, patient tags, and level.
    Returns a dict with ranked modules and rationales.
    """
    # Load region disease prevalence
    region_file = os.path.join(os.path.dirname(__file__), '../../data/regions.csv')
    df = pd.read_csv(region_file)
    row = df[df['region'] == region]
    prevalence = {}
    if not row.empty:
        for pair in row.iloc[0]['disease_prevalence'].split(','):
            k, v = pair.split(':')
            prevalence[k.strip()] = float(v)
    # Map tags to modules
    tag_modules = set()
    for tag in patient_tags:
        mod = SYMPTOM_TO_MODULE.get(tag.lower())
        if mod:
            tag_modules.add(mod)
    # Score modules: region prevalence + tag match + level
    ranked = []
    for mod in MODULES:
        score = 0.0
        rationale = []
        # Region prevalence
        for disease in prevalence:
            if disease.lower() in mod.lower():
                score += prevalence[disease] * 2
                rationale.append(f"High {disease} prevalence in {region}")
        # Tag match
        if mod in tag_modules:
            score += 1.5
            rationale.append(f"Relevant to patient tags: {', '.join(patient_tags)}")
        # Level (stub: boost for 'advanced')
        if level == 'advanced':
            score += 0.5
            rationale.append("Advanced level selected")
        ranked.append((mod, score, rationale))
    ranked.sort(key=lambda x: -x[1])
    modules = [{"module": m, "score": round(s,2), "rationales": r} for m, s, r in ranked if s > 0]
    # Adaptive next steps
    next_steps = [m["module"] for m in modules[:2]]
    return {"modules": modules, "next_steps": next_steps, "region": region, "patient_tags": patient_tags, "level": level}
