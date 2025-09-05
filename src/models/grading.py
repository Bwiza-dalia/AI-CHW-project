from typing import Dict, List
from .utils import normalize_text, get_text_embedding, cosine_similarity, expand_with_synonyms
import numpy as np
import os
import re

# Example key concepts for demo (in real use, extract from reference)
EN_KEYWORDS = ["malaria", "prevention", "mosquito", "bed net", "symptoms", "treatment"]
KI_KEYWORDS = ["malariya", "kwirinda", "umubu", "uburiri", "ibimenyetso", "ubuvuzi"]

# Improved feedback templates
FEEDBACK_TEMPLATES = {
    'en': {
        5: "Excellent answer. You've covered the key points clearly.",
        4: "Good answer. Consider adding a few more details.",
        3: "Fair answer. Try to include more key terms and details.",
        2: "Your answer needs more explanation and examples.",
        1: "This answer is too brief or off-topic. Please review the material.",
        0: "This answer is completely off-topic or shows no understanding. Please study the material."
    },
    'ki': {
        5: "Igisubizo cyiza cyane. Wasobanuye ingingo zose neza.",
        4: "Igisubizo cyiza. Wagerageza kongeramo andi makuru.",
        3: "Hari ibyo wasobanura neza kurushaho.",
        2: "Ibisubizo bikwiye kuba birambuye kurushaho.",
        1: "Igisubizo ni kigufi cyangwa kidahuye n'ibikenewe.",
        0: "Igisubizo ntihuye n'ibikenewe cyangwa nta bumenyi buhabwa. Soma ibisobanuro."
    }
}

def is_off_topic_answer(user_answer: str, reference_answer: str, lang: str) -> bool:
    """Detect if user answer is completely off-topic or shows no understanding."""
    # Check for "I don't know" type responses
    dont_know_patterns = [
        r"i don't know", r"i don't understand", r"no idea", r"not sure",
        r"simbyumva", r"ntabumenyi", r"nta bumenyi", r"nta kintu"
    ]
    
    user_lower = user_answer.lower()
    for pattern in dont_know_patterns:
        if re.search(pattern, user_lower):
            return True
    
    # Check if answer is extremely short compared to reference (only if it's really minimal)
    if len(user_answer.split()) < 3:  # Only mark as off-topic if less than 3 words
        return True
    
    # Don't use strict semantic similarity threshold - let the ML model handle scoring
    # This was causing everything to get score 0
    return False

def extract_key_concepts(text: str, lang: str) -> List[str]:
    """Extracts key concepts by keyword/synonym match."""
    if lang.lower().startswith("ki"):
        keywords = KI_KEYWORDS
    else:
        keywords = EN_KEYWORDS
    found = set()
    text_norm = text.lower()
    for k in keywords:
        for syn in expand_with_synonyms(k, lang):
            if syn in text_norm:
                found.add(k)
    return list(found)

def map_similarity_to_score(sim: float) -> int:
    """Map cosine similarity to 0-5 score (calibrated thresholds)."""
    if sim >= 0.85:
        return 5
    elif sim >= 0.7:
        return 4
    elif sim >= 0.5:
        return 3
    elif sim >= 0.3:
        return 2
    elif sim >= 0.15:  # Lowered threshold to get more 1s
        return 1
    else:
        return 0

def predict_score_ml(features: Dict[str, float]) -> float:
    """
    Predict score using trained Random Forest model if available.
    Returns a float score (0-5), or None if model not found.
    """
    try:
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), '../../data/grading_rf_model.pkl')
        rf = joblib.load(model_path)
        X = np.array([[features[k] for k in sorted(features)]])
        pred = rf.predict(X)[0]
        return float(np.clip(pred, 0, 5))
    except Exception:
        return None

def calculate_grading_confidence(features: Dict[str, float], similarity: float, score: int, ml_score: float = None) -> float:
    """
    Calculate confidence score for grading based on semantic understanding.
    Returns a confidence score between 0.0 and 1.0.
    """
    confidence_factors = []
    
    # Factor 1: Semantic similarity (most important - 50%)
    confidence_factors.append(similarity)
    
    # Factor 2: ML model agreement (if available - 20%)
    if ml_score is not None:
        similarity_score = map_similarity_to_score(similarity)
        score_agreement = 1.0 - abs(ml_score - similarity_score) / 5.0
        confidence_factors.append(score_agreement)
    
    # Factor 3: Semantic feature strength (30%)
    # Focus more on semantic features rather than keyword matching
    semantic_strength = (
        features.get('emb_sim', 0) * 0.5 +  # Embedding similarity (most important)
        features.get('tfidf_sim', 0) * 0.3 +  # TF-IDF similarity
        features.get('jaccard_sim', 0) * 0.2   # Jaccard similarity
    )
    confidence_factors.append(semantic_strength)
    
    # Factor 4: Answer completeness (20%)
    # Check if answer covers the main concepts semantically
    len_ratio = features.get('len_ratio', 0)
    completeness_factor = min(len_ratio * 2, 1.0)  # Normalize length ratio
    confidence_factors.append(completeness_factor)
    
    # Calculate weighted average
    if ml_score is not None:
        weights = [0.5, 0.2, 0.3, 0.0]  # ML model available
    else:
        weights = [0.6, 0.0, 0.3, 0.1]  # No ML model
    
    confidence = sum(w * f for w, f in zip(weights, confidence_factors))
    return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1

def grade_answer(question: str, reference_answer: str, user_answer: str, lang: str) -> Dict:
    """
    Grades a descriptive answer using ML model if available, else rule-based fallback.
    Returns a dict: {score_0_to_5, similarity, strengths, gaps, suggestions, confidence, lang}
    """
    # Check if answer is completely off-topic
    if is_off_topic_answer(user_answer, reference_answer, lang):
        feedback = {
            'score_0_to_5': 0,
            'similarity': 0.0,
            'strengths': [],
            'gaps': ["Complete misunderstanding or off-topic response"],
            'suggestions': [FEEDBACK_TEMPLATES[lang if lang in FEEDBACK_TEMPLATES else "en"][0]],
            'confidence': 0.95,  # High confidence for clear off-topic answers
            'lang': lang
        }
        return feedback
    
    # Feature extraction
    features = extract_features(question, reference_answer, user_answer, lang)
    # Try ML model
    ml_score = predict_score_ml(features)
    # Embedding similarity for feedback
    ref_norm = normalize_text(reference_answer, lang)
    user_norm = normalize_text(user_answer, lang)
    ref_emb = get_text_embedding(ref_norm, lang)
    user_emb = get_text_embedding(user_norm, lang)
    similarity = cosine_similarity(ref_emb, user_emb)
    
    # Determine final score based on semantic understanding
    if ml_score is not None:
        score = int(round(ml_score))
    else:
        score = map_similarity_to_score(similarity)
    
    # Calculate confidence
    confidence = calculate_grading_confidence(features, similarity, score, ml_score)
    
    # Get feedback based on score and semantic understanding
    fb = FEEDBACK_TEMPLATES[lang if lang in FEEDBACK_TEMPLATES else "en"]
    main_feedback = fb.get(score, fb[1])
    
    # Generate semantic-based suggestions
    suggestions = [main_feedback]
    
    # Add semantic understanding feedback
    if similarity < 0.3:
        suggestions.append("Your answer doesn't seem to address the main concepts. Try to focus on the key ideas.")
    elif similarity < 0.6:
        suggestions.append("Your answer touches on some concepts but could be more comprehensive.")
    
    if score < 3:
        suggestions.append("Review the lesson material and try to provide more detailed answers.")
    
    # Add confidence-based feedback
    if confidence < 0.5:
        suggestions.append("Note: The grading confidence is low. Consider reviewing your answer for clarity.")
    
    feedback = {
        'score_0_to_5': score,
        'similarity': round(similarity, 3),
        'suggestions': suggestions,
        'confidence': round(confidence, 3),
        'lang': lang
    }
    return feedback

def extract_features(question: str, reference_answer: str, user_answer: str, lang: str) -> Dict[str, float]:
    """
    Extracts features for grading:
    - TF-IDF cosine similarity
    - Jaccard similarity
    - Keyword density
    - Embedding similarity
    - Answer length ratio
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Normalize
    ref_norm = normalize_text(reference_answer, lang)
    user_norm = normalize_text(user_answer, lang)
    # TF-IDF similarity
    try:
        vec = TfidfVectorizer()
        X = vec.fit_transform([ref_norm, user_norm])
        tfidf_sim = float((X[0] @ X[1].T).A[0][0])
    except Exception:
        tfidf_sim = 0.0
    # Jaccard similarity
    ref_set = set(ref_norm.split())
    user_set = set(user_norm.split())
    intersection = len(ref_set & user_set)
    union = len(ref_set | user_set)
    jaccard_sim = intersection / union if union else 0.0
    # Keyword density
    if lang.lower().startswith("ki"):
        keywords = [k for k in ["malariya", "kwirinda", "umubu", "uburiri", "ibimenyetso", "ubuvuzi"]]
    else:
        keywords = [k for k in ["malaria", "prevention", "mosquito", "bed net", "symptoms", "treatment"]]
    keyword_count = sum(1 for k in keywords if k in user_norm)
    keyword_density = keyword_count / len(keywords) if keywords else 0.0
    # Embedding similarity
    ref_emb = get_text_embedding(ref_norm, lang)
    user_emb = get_text_embedding(user_norm, lang)
    emb_sim = cosine_similarity(ref_emb, user_emb)
    # Answer length ratio
    ref_len = len(ref_norm.split())
    user_len = len(user_norm.split())
    len_ratio = user_len / ref_len if ref_len else 0.0
    return {
        'tfidf_sim': tfidf_sim,
        'jaccard_sim': jaccard_sim,
        'keyword_density': keyword_density,
        'emb_sim': emb_sim,
        'len_ratio': len_ratio
    }
