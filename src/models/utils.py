from typing import List
import re

# Minimal stopword lists for demo
EN_STOPWORDS = set("the a an and or but if in on with for to of is are was were be by as at from that this it".split())
KI_STOPWORDS = set(["na", "ku", "ni", "ya", "mu", "bya", "kuri", "cyangwa", "ariko", "ubwo", "kandi", "buri", "iyo", "mu", "kuva", "uko", "icyo", "ibi", "iri", "uyu", "bya"])

KI_SYNONYMS = {
    "malariya": ["malaria"],
    "kwirinda": ["gukumira", "kurinda"],
    "umubu": ["umubu", "imibu"],
    "uburiri": ["uburiri", "igitanda"],
    "ibimenyetso": ["ibimenyetso", "ikimenyetso"],
    "ubuvuzi": ["ubuvuzi", "kwivuza"]
}

def normalize_text(text: str, lang: str) -> str:
    """Lowercase, remove stopwords (per lang), basic normalization."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    if lang.lower().startswith("ki"):
        stopwords = KI_STOPWORDS
    else:
        stopwords = EN_STOPWORDS
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

def detect_language(text: str) -> str:
    """Detects language (EN/KI) from text. Simple heuristic for now."""
    if any(word in text.lower() for word in ["umwana", "ubuzima", "indwara", "muganga", "kwivuza"]):
        return "ki"
    return "en"

def get_text_embedding(text: str, lang: str) -> List[float]:
    """Returns embedding for text using sentence-transformers if available, else TF-IDF vector."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model.encode([text])[0].tolist()
    except Exception:
        # Fallback: TF-IDF with consistent vocabulary
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Use a larger vocabulary size and ensure consistent dimensions
            vec = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9)
            # Fit on a sample of text to build vocabulary
            sample_texts = [
                "glaucoma symptoms eye pressure optic nerve",
                "diabetes blood sugar insulin treatment",
                "malaria prevention mosquito bed net",
                "hypertension blood pressure heart disease",
                "diarrhea dehydration rehydration treatment"
            ]
            vec.fit(sample_texts)
            X = vec.transform([text])
            return X.toarray()[0].tolist()
        except Exception:
            # Fallback: zeros with consistent dimension
            return [0.0] * 100

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    v1, v2 = np.array(vec1), np.array(vec2)
    if v1.shape != v2.shape or v1.shape[0] == 0:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def expand_with_synonyms(word: str, lang: str) -> List[str]:
    """
    Expand a word with its synonyms using WordNet for English, or a simple mapping for Kinyarwanda.
    Returns a list of synonyms including the original word.
    """
    if lang.lower().startswith("ki"):
        return list(set([word] + KI_SYNONYMS.get(word, [])))
    else:
        try:
            from nltk.corpus import wordnet as wn
            syns = set([word])
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    syns.add(lemma.name().replace('_', ' '))
            return list(syns)
        except Exception:
            return [word]
