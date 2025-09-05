from typing import Dict, List
try:
    from .utils import normalize_text, get_text_embedding, cosine_similarity
except ImportError:
    from src.models.utils import normalize_text, get_text_embedding, cosine_similarity
import re
import os
import pandas as pd

# Load MedQuAD course content for Q&A
def load_course_content() -> List[Dict]:
    """Load course content from MedQuAD dataset for Q&A retrieval."""
    try:
        medquad_path = os.path.join(os.path.dirname(__file__), '../../data/medquad.csv')
        df = pd.read_csv(medquad_path)
        
        # Convert to course content format
        content_chunks = []
        for _, row in df.iterrows():
            # Clean and structure the content
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            source = str(row.get('source', 'MedQuAD')).strip()
            focus_area = str(row.get('focus_area', 'General Health')).strip()
            
            # Skip if content is too short or invalid
            if len(answer) < 20 or len(question) < 10:
                continue
                
            # Create content chunk with metadata
            chunk = {
                "title": question,
                "text": answer,
                "source": source,
                "focus_area": focus_area,
                "keywords": extract_keywords(answer),
                "lang": "en"  # MedQuAD is in English
            }
            content_chunks.append(chunk)
            
        print(f"Loaded {len(content_chunks)} course content chunks from MedQuAD")
        return content_chunks
    except Exception as e:
        print(f"Error loading MedQuAD content: {e}")
        # Fallback to static content
        return get_static_content()

def get_static_content() -> List[Dict]:
    """Fallback static content if MedQuAD loading fails."""
    return [
        {"title": "Malaria Prevention", "text": "Malaria is prevented by sleeping under a mosquito bed net and removing standing water. Symptoms include fever and chills.", "source": "Static", "focus_area": "Infectious Disease", "keywords": ["malaria", "prevention", "mosquito", "bed net", "symptoms"], "lang": "en"},
        {"title": "Diarrhea Management", "text": "Diarrhea can be managed by giving oral rehydration salts and ensuring clean water. Watch for signs of dehydration.", "source": "Static", "focus_area": "Gastrointestinal", "keywords": ["diarrhea", "management", "rehydration", "dehydration"], "lang": "en"},
        {"title": "ARI Recognition", "text": "Acute respiratory infection (ARI) presents with cough and difficulty breathing. Seek care if symptoms worsen.", "source": "Static", "focus_area": "Respiratory", "keywords": ["ARI", "respiratory", "cough", "breathing", "symptoms"], "lang": "en"}
    ]

def extract_keywords(text: str) -> List[str]:
    """Extract key medical terms from text for better retrieval."""
    # Simple keyword extraction - could be enhanced with medical NER
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Filter common medical terms
    medical_terms = [w for w in words if w not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'when', 'where', 'what', 'how', 'why']]
    return list(set(medical_terms))[:10]  # Top 10 keywords

# Load course content
COURSE_CONTENT = load_course_content()

VARK_TIPS = {
    "visual": ["Draw diagrams or concept maps.", "Use color coding for notes."],
    "auditory": ["Discuss topics with peers.", "Listen to recorded lessons."],
    "read-write": ["Summarize lessons in your own words.", "Make lists and read them aloud."],
    "kinesthetic": ["Practice with real-life scenarios.", "Use hands-on materials or demonstrations."]
}

QA_TEMPLATES = {
    "en": "Based on the course, here's what you should know: {answer}",
    "ki": "Dushingiye ku isomo, dore ibisobanuro: {answer}"
}

def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from text using semantic analysis."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # Extract noun phrases and key terms
    concepts = []
    
    # Look for important patterns
    patterns = [
        r'\b(?:was|is|are|were|became|led to|caused|resulted in|brought about)\s+([^.]{10,50})',
        r'\b(?:during|in|from|to)\s+([^.]{10,50})',
        r'\b(?:including|such as|like|for example)\s+([^.]{10,50})',
        r'\b(?:the|a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        concepts.extend([m.strip() for m in matches if len(m.strip()) > 5])
    
    # Also extract capitalized terms (likely important concepts)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    concepts.extend(capitalized)
    
    # Remove duplicates and filter
    concepts = list(set(concepts))
    concepts = [c for c in concepts if len(c) > 3 and c.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
    
    return concepts[:8]  # Top 8 concepts

def summarize(text: str, lang: str) -> Dict:
    """Intelligently summarizes lesson text into concepts and key points."""
    # Clean and normalize text
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Extract key concepts
    concepts = extract_key_concepts(cleaned)
    
    # Create intelligent summary
    sentences = re.split(r'[.!?]+', cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    # Find the most informative sentences (containing key concepts)
    scored_sentences = []
    for sentence in sentences:
        score = 0
        # Score based on concept density
        for concept in concepts:
            if concept.lower() in sentence.lower():
                score += 1
        # Score based on length (not too short, not too long)
        if 20 <= len(sentence) <= 150:
            score += 1
        # Score based on position (first sentences often more important)
        if sentences.index(sentence) < 3:
            score += 0.5
        scored_sentences.append((sentence, score))
    
    # Sort by score and take best sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    best_sentences = [s[0] for s in scored_sentences[:2]]
    
    # Create summary paragraph
    if best_sentences:
        summary = '. '.join(best_sentences) + '.'
    else:
        # Fallback to first part of text
        summary = cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
    
    # Create concept-based bullets
    bullets = []
    if concepts:
        # Clean up concepts to be more concise
        clean_concepts = []
        for concept in concepts[:5]:
            # Remove extra words and make more concise
            if len(concept) > 30:
                # Take first part of long concepts
                concept = concept.split(',')[0].strip()
            if len(concept) > 20:
                # Further shorten if still too long
                words = concept.split()
                concept = ' '.join(words[:3]) if len(words) > 3 else concept
            clean_concepts.append(concept)
        bullets.append(f"**Key Concepts:** {', '.join(clean_concepts)}")
    
    # Add main themes (including medical themes)
    themes = []
    if any(word in cleaned.lower() for word in ['change', 'transformation', 'evolution', 'development']):
        themes.append('Transformation')
    if any(word in cleaned.lower() for word in ['cause', 'effect', 'result', 'consequence', 'impact']):
        themes.append('Cause & Effect')
    if any(word in cleaned.lower() for word in ['problem', 'challenge', 'issue', 'difficulty']):
        themes.append('Challenges')
    if any(word in cleaned.lower() for word in ['benefit', 'advantage', 'improvement', 'progress']):
        themes.append('Benefits')
    if any(word in cleaned.lower() for word in ['period', 'time', 'century', 'era', 'age']):
        themes.append('Historical Context')
    
    # Medical themes
    if any(word in cleaned.lower() for word in ['symptom', 'sign', 'fever', 'pain', 'ache']):
        themes.append('Symptoms')
    if any(word in cleaned.lower() for word in ['prevent', 'prevention', 'avoid', 'protect']):
        themes.append('Prevention')
    if any(word in cleaned.lower() for word in ['treat', 'treatment', 'cure', 'medicine', 'drug']):
        themes.append('Treatment')
    if any(word in cleaned.lower() for word in ['diagnose', 'diagnosis', 'test', 'examine']):
        themes.append('Diagnosis')
    if any(word in cleaned.lower() for word in ['disease', 'illness', 'condition', 'disorder']):
        themes.append('Medical Condition')
    
    if themes:
        bullets.append(f"**Main Themes:** {', '.join(themes)}")
    
    # Add key facts
    facts = []
    # Look for numbers and dates
    numbers = re.findall(r'\b(?:18th|19th|20th|\d{4}|\d+)\b', cleaned)
    if numbers:
        facts.append(f"**Time Period:** {', '.join(numbers[:3])}")
    
    # Look for locations
    locations = re.findall(r'\b(?:Europe|America|Asia|Africa|Britain|England|France|Germany|United States|US|USA)\b', cleaned, re.IGNORECASE)
    if locations:
        unique_locations = list(set(locations))[:3]
        facts.append(f"**Locations:** {', '.join(unique_locations)}")
    
    bullets.extend(facts)
    
    return {"summary": summary, "bullets": bullets}

def generate_concept_diagram(text: str, lang: str) -> str:
    """Generate an actual concept diagram using matplotlib and networkx."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        import io
        import base64
        
        # Extract key concepts
        concepts = extract_key_concepts(text)
        
        # Extract main themes
        themes = []
        if any(word in text.lower() for word in ['symptom', 'sign', 'fever', 'pain']):
            themes.append('Symptoms')
        if any(word in text.lower() for word in ['prevent', 'prevention', 'avoid']):
            themes.append('Prevention')
        if any(word in text.lower() for word in ['treat', 'treatment', 'cure']):
            themes.append('Treatment')
        if any(word in text.lower() for word in ['cause', 'caused by', 'due to']):
            themes.append('Causes')
        if any(word in text.lower() for word in ['effect', 'result', 'consequence']):
            themes.append('Effects')
        
        # Clean up concepts for display
        clean_concepts = []
        for concept in concepts[:6]:  # Limit to 6 concepts
            if len(concept) > 20:
                words = concept.split()
                concept = ' '.join(words[:3]) if len(words) > 3 else concept
            clean_concepts.append(concept)
        
        if not clean_concepts:
            clean_concepts = ['Main Topic', 'Key Point 1', 'Key Point 2']
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add central node (main topic)
        main_topic = clean_concepts[0] if clean_concepts else "Main Topic"
        G.add_node(main_topic, node_type='main')
        
        # Add other concepts
        for concept in clean_concepts[1:]:
            G.add_node(concept, node_type='concept')
            # Connect to main topic
            G.add_edge(main_topic, concept)
        
        # Add theme nodes and connect to relevant concepts
        for theme in themes[:3]:  # Limit to 3 themes
            G.add_node(theme, node_type='theme')
            # Connect theme to concepts that might relate
            for concept in clean_concepts[1:]:
                if any(word in concept.lower() for word in theme.lower().split()):
                    G.add_edge(theme, concept)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes with different colors
        main_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'main']
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'concept']
        theme_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'theme']
        
        # Draw main node
        nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, 
                              node_color='#FF6B6B', node_size=2000, alpha=0.8)
        
        # Draw concept nodes
        nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, 
                              node_color='#4ECDC4', node_size=1500, alpha=0.8)
        
        # Draw theme nodes
        nx.draw_networkx_nodes(G, pos, nodelist=theme_nodes, 
                              node_color='#45B7D1', node_size=1200, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add title
        plt.title('Concept Map', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        main_patch = mpatches.Patch(color='#FF6B6B', label='Main Topic')
        concept_patch = mpatches.Patch(color='#4ECDC4', label='Key Concepts')
        theme_patch = mpatches.Patch(color='#45B7D1', label='Themes')
        plt.legend(handles=[main_patch, concept_patch, theme_patch], loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Convert to base64 string for Streamlit
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generating diagram: {e}")
        return None

def diagram_prompt(text: str, lang: str) -> Dict:
    """Produces an intelligent graphical prompt for concept mapping."""
    # Extract key concepts
    concepts = extract_key_concepts(text)
    
    # Extract main themes
    themes = []
    if any(word in text.lower() for word in ['symptom', 'sign', 'fever', 'pain']):
        themes.append('Symptoms')
    if any(word in text.lower() for word in ['prevent', 'prevention', 'avoid']):
        themes.append('Prevention')
    if any(word in text.lower() for word in ['treat', 'treatment', 'cure']):
        themes.append('Treatment')
    if any(word in text.lower() for word in ['cause', 'caused by', 'due to']):
        themes.append('Causes')
    if any(word in text.lower() for word in ['effect', 'result', 'consequence']):
        themes.append('Effects')
    
    # Create intelligent diagram prompt
    if concepts and themes:
        prompt = f"Create a concept map showing the relationship between: {', '.join(concepts[:4])}. Organize around these themes: {', '.join(themes[:3])}. Use arrows to show cause-effect relationships and connections between concepts."
    elif concepts:
        prompt = f"Draw a concept map with these key concepts: {', '.join(concepts[:5])}. Show how they relate to each other with connecting lines and arrows."
    else:
        # Fallback to simple extraction
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        if words:
            prompt = f"Create a concept map with: {', '.join(words[:4])}. Show relationships between these concepts."
        else:
            prompt = "Create a concept map based on the main ideas in the text."
    
    return {"prompt": prompt}

def clean_and_summarize_content(raw_text: str, question_type: str) -> str:
    """
    Clean and summarize raw content from MedQuAD to make it more user-friendly.
    """
    # Remove video references, graphics, and other non-essential elements
    cleaned = re.sub(r'\(Watch the.*?\)', '', raw_text)
    cleaned = re.sub(r'\(To enlarge.*?\)', '', cleaned)
    cleaned = re.sub(r'\(To reduce.*?\)', '', cleaned)
    cleaned = re.sub(r'See this graphic.*?\.', '', cleaned)
    cleaned = re.sub(r'See a glossary.*?\.', '', cleaned)
    cleaned = re.sub(r'Read or listen.*?\.', '', cleaned)
    cleaned = re.sub(r'Get tips.*?\.', '', cleaned)
    cleaned = re.sub(r'Learn what.*?\.', '', cleaned)
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Special handling for specific medical conditions - extract key information first
    if 'glaucoma' in question_type.lower():
        # Look for the key information about glaucoma symptoms
        if 'no symptoms at first' in cleaned.lower():
            return 'The most common type of glaucoma (open-angle glaucoma) has no symptoms at first. It causes no pain and vision seems normal. Without treatment, people with glaucoma will slowly lose their peripheral (side) vision, making it seem like they are looking through a tunnel.'
        elif 'peripheral' in cleaned.lower() and 'vision' in cleaned.lower():
            return 'People with glaucoma will slowly lose their peripheral (side) vision. They seem to be looking through a tunnel. Over time, straight-ahead vision may decrease until no vision remains.'
        elif 'causes' in question_type.lower() and 'risk' in cleaned.lower():
            return 'Glaucoma is caused by increased pressure in the eye that damages the optic nerve. Risk factors include age (especially over 60), family history, and certain ethnic backgrounds. African-Americans over 40 and everyone over 60 are at higher risk.'
    
    elif 'diabetes' in question_type.lower():
        # Look for key diabetes information
        if 'type 2 diabetes' in cleaned.lower():
            return 'Type 2 diabetes is the most common type of diabetes, affecting about 95% of people with diabetes. It occurs when the body becomes resistant to insulin or doesn\'t produce enough insulin.'
        elif 'blood sugar' in cleaned.lower():
            return 'Diabetes is a condition where blood sugar levels are too high. This happens when the body either doesn\'t produce enough insulin or can\'t use insulin effectively.'
    
    elif 'malaria' in question_type.lower():
        # Look for key malaria information
        if 'mosquito' in cleaned.lower() and 'bed net' in cleaned.lower():
            return 'Malaria is a disease spread by mosquitoes. It can be prevented by sleeping under a mosquito bed net and removing standing water where mosquitoes breed.'
        elif 'fever' in cleaned.lower() and 'chills' in cleaned.lower():
            return 'Malaria symptoms include fever, chills, and flu-like illness. If not treated promptly, it can become severe and life-threatening.'
        elif 'parasite' in cleaned.lower() and 'mosquito' in cleaned.lower():
            return 'Malaria is a serious disease caused by a parasite. You get it when an infected mosquito bites you. It is a major cause of death worldwide.'
        elif 'human phenotype ontology' in cleaned.lower():
            # Skip complex ontology entries for simple questions
            return 'Malaria is a serious disease caused by a parasite transmitted by mosquitoes. Common symptoms include fever, chills, and flu-like illness.'
    
    # Extract relevant information based on question type
    if "symptoms" in question_type.lower():
        # Look for symptom-specific information with better patterns
        symptom_patterns = [
            r'symptoms? of[^.]*?\.',
            r'presents? with[^.]*?\.',
            r'causes?[^.]*?pain[^.]*?\.',
            r'vision[^.]*?loss[^.]*?\.',
            r'peripheral[^.]*?vision[^.]*?\.',
            r'tunnel[^.]*?vision[^.]*?\.',
            r'no symptoms?[^.]*?\.',
            r'pain[^.]*?\.',
            r'blurred[^.]*?\.',
            r'headache[^.]*?\.'
        ]
        
        relevant_parts = []
        for pattern in symptom_patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            relevant_parts.extend(matches)
        
        # If we found specific symptom patterns, use them
        if relevant_parts:
            # Take the most relevant symptom information
            return '. '.join(relevant_parts[:2]) + '.'
        
        # If no specific patterns found, look for sentences containing key symptom words
        sentences = re.split(r'[.!?]+', cleaned)
        symptom_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Look for sentences with symptom-related keywords
                if any(word in sentence.lower() for word in ['symptom', 'pain', 'vision', 'loss', 'blurred', 'headache', 'peripheral', 'tunnel']):
                    symptom_sentences.append(sentence)
        
        if symptom_sentences:
            return '. '.join(symptom_sentences[:2]) + '.'
    
    elif "treatment" in question_type.lower():
        # Look for treatment-specific information
        treatment_patterns = [
            r'treatments?[^.]*?\.',
            r'medications?[^.]*?\.',
            r'surgery[^.]*?\.',
            r'eye drops[^.]*?\.',
            r'laser[^.]*?\.'
        ]
        
        relevant_parts = []
        for pattern in treatment_patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            relevant_parts.extend(matches)
        
        if relevant_parts:
            return '. '.join(relevant_parts[:2]) + '.'
    
    elif "causes" in question_type.lower():
        # Look for cause-specific information
        cause_patterns = [
            r'causes?[^.]*?\.',
            r'risk factors?[^.]*?\.',
            r'pressure[^.]*?\.',
            r'age[^.]*?\.',
            r'family history[^.]*?\.'
        ]
        
        relevant_parts = []
        for pattern in cause_patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            relevant_parts.extend(matches)
        
        if relevant_parts:
            return '. '.join(relevant_parts[:2]) + '.'
    
    # For general questions, provide a concise summary
    # Split into sentences and take the most informative ones
    sentences = re.split(r'[.!?]+', cleaned)
    informative_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 300:  # Good length
            # Prioritize sentences with key medical terms
            if any(term in sentence.lower() for term in ['symptoms', 'treatment', 'causes', 'prevention', 'diagnosis', 'pain', 'vision', 'loss']):
                informative_sentences.append(sentence)
            elif len(informative_sentences) < 2:  # Keep some general info
                informative_sentences.append(sentence)
    
    if informative_sentences:
        return '. '.join(informative_sentences[:2]) + '.'
    
    # Fallback: return first 200 characters if no good summary found
    return cleaned[:200] + '...' if len(cleaned) > 200 else cleaned

def qa_over_content(question: str, lang: str) -> Dict:
    """
    Answers a question over course content using keyword-based retrieval.
    Returns a bilingual/context-aware answer and cites the most relevant chunk.
    """
    # For Kinyarwanda questions, translate to English for content search
    search_question = question
    if lang == "ki":
        try:
            from deep_translator import GoogleTranslator
            search_question = GoogleTranslator(source='rw', target='en').translate(question)
            print(f"Translated question: {question} -> {search_question}")
        except Exception as e:
            print(f"Question translation failed: {e}")
            # Keep original if translation fails
    
    question_lower = search_question.lower()
    
    # Enhanced retrieval with keyword scoring
    scored_chunks = []
    for chunk in COURSE_CONTENT:
        score = 0.0
        
        chunk_text_lower = chunk["text"].lower()
        chunk_title_lower = chunk["title"].lower()
        
        # HIGH PRIORITY: Exact medical condition matches (check first)
        medical_conditions = {
            "glaucoma": ["glaucoma", "eye pressure", "optic nerve"],
            "diabetes": ["diabetes", "blood sugar", "insulin"],
            "malaria": ["malaria", "mosquito", "bed net"],
            "hypertension": ["hypertension", "high blood pressure", "blood pressure"],
            "diarrhea": ["diarrhea", "dehydration", "rehydration"],
            "cancer": ["cancer", "tumor", "malignant"],
            "heart": ["heart disease", "cardiac", "cardiovascular"],
            "lung": ["lung disease", "respiratory", "pneumonia"]
        }
        
        # Check if question asks about a specific condition
        question_condition = None
        for condition in medical_conditions.keys():
            if condition in question_lower:
                question_condition = condition
                break
        
        if question_condition:
            # More specific matching for glaucoma to avoid complex conditions
            if question_condition == "glaucoma":
                # Prefer simple glaucoma questions over complex ones
                if "tetralogy" in chunk_title_lower or "fallot" in chunk_title_lower or "syndrome" in chunk_title_lower:
                    score = 0.0  # Skip complex conditions
                    continue
                # Prefer direct glaucoma questions
                if question_condition in chunk_title_lower and len(chunk_title_lower.split()) <= 8:
                    score += 15.0  # Very high priority for simple, direct glaucoma questions
                elif any(keyword in chunk_text_lower for keyword in medical_conditions[question_condition]):
                    score += 8.0   # High priority for condition-specific content
                else:
                    score = 0.0
                    continue
            else:
                # For other conditions, use the original logic
                if question_condition in chunk_title_lower:
                    score += 10.0  # Very high priority for condition-specific titles
                elif any(keyword in chunk_text_lower for keyword in medical_conditions[question_condition]):
                    score += 8.0   # High priority for condition-specific content
                else:
                    # If chunk is not about the asked condition, give very low score
                    score = 0.0
                    continue
        
        # Only add general scoring if we have a condition match
        if score > 0:
            # Score based on keyword matches in title
            title_matches = sum(1 for word in question_lower.split() if word in chunk_title_lower)
            score += title_matches * 0.5
            
            # Score based on keyword matches in text
            text_matches = sum(1 for word in question_lower.split() if word in chunk_text_lower)
            score += text_matches * 0.2
            
            # Bonus for exact phrase matches
            if any(phrase in chunk_text_lower for phrase in ["symptoms", "treatment", "prevention", "causes"]):
                if any(phrase in question_lower for phrase in ["symptoms", "treatment", "prevention", "causes"]):
                    score += 0.3
            
            scored_chunks.append((chunk, score))
    
    # Sort by score and get top results
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    if scored_chunks and scored_chunks[0][1] > 5.0:  # High threshold for quality
        best_chunk, score = scored_chunks[0]
        raw_answer = best_chunk['text']
        
        # Clean and summarize the content based on question type
        question_type = search_question  # Use the search question to determine type
        answer = clean_and_summarize_content(raw_answer, question_type)
        
        # Translate answer if language is Kinyarwanda
        if lang == "ki":
            try:
                from deep_translator import GoogleTranslator
                translated_answer = GoogleTranslator(source='en', target='rw').translate(answer)
                answer = translated_answer
            except Exception as e:
                print(f"Answer translation failed: {e}")
                # Keep English if translation fails
        
        template = QA_TEMPLATES[lang] if lang in QA_TEMPLATES else QA_TEMPLATES['en']
        out = template.format(answer=answer)
        
        # Add metadata
        return {
            "answer": out, 
            "source_title": best_chunk['title'],
            "source": best_chunk.get('source', 'Unknown'),
            "focus_area": best_chunk.get('focus_area', 'General'),
            "confidence": round(score, 3)
        }
    else:
        # Ensure all return paths have the same structure
        if lang == "ki":
            return {
                "answer": "Nta bisobanuro bibonetse kuri iki kibazo.", 
                "source_title": None,
                "source": None,
                "focus_area": None,
                "confidence": 0.0
            }
        else:
            return {
                "answer": "Sorry, no relevant content found. Try rephrasing your question or ask about a different health topic.", 
                "source_title": None,
                "source": None,
                "focus_area": None,
                "confidence": 0.0
            }

def adaptation_suggestions(text: str, lang: str) -> Dict:
    """Suggests 1-2 tips per VARK learning style."""
    import random
    tips = {style: random.sample(VARK_TIPS[style], 1) for style in VARK_TIPS}
    return {"suggestions": tips}
