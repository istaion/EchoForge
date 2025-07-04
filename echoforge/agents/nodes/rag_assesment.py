"""Nœud d'évaluation du besoin de recherche RAG."""

import re
from typing import List, Optional
from ..state.character_state import CharacterState


def assess_rag_need(state: CharacterState) -> CharacterState:
    """
    Évalue si une recherche RAG est nécessaire pour répondre à la requête.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec l'évaluation RAG
    """
    state["processing_steps"].append("rag_assessment")
    
    message = state["parsed_message"]
    intent = state["message_intent"]
    
    # Analyse détaillée du besoin RAG
    rag_analysis = _detailed_rag_analysis(message, intent, state)
    
    # Mise à jour de l'état
    state["needs_rag_search"] = rag_analysis["needs_rag"]
    state["rag_query"] = rag_analysis["query"]
    
    # Debug info
    state["debug_info"]["rag_assessment"] = rag_analysis
    
    return state


def _detailed_rag_analysis(message: str, intent: str, state: CharacterState) -> dict:
    """Analyse détaillée du besoin de recherche RAG."""
    message_lower = message.lower()
    
    # === INDICATEURS DE BESOIN RAG ===
    
    # 1. Mots-clés nécessitant une recherche de connaissances
    knowledge_keywords = [
        "histoire", "passé", "avant", "autrefois", "jadis",
        "secret", "mystère", "caché", "confidentiel",
        "relation", "ami", "ennemi", "famille",
        "événement", "incident", "accident", "guerre",
        "souvenir", "mémoire", "rappelle",
        "pourquoi", "comment", "raison", "cause",
        "qui est", "qu'est-ce que", "où se trouve",
        "origine", "création", "fondation",
        "tradition", "coutume", "rituel"
    ]
    
    # 2. Questions spécifiques
    question_patterns = [
        r"qui (?:est|était|a été)",
        r"qu'est-ce que",
        r"pourquoi (?:est-ce que|as-tu|avez-vous)",
        r"comment (?:est-ce que|as-tu|avez-vous)",
        r"où (?:est|se trouve|as-tu)",
        r"quand (?:est-ce que|as-tu|avez-vous)",
        r"raconte(?:-moi)? (?:l'histoire|le passé|les événements)",
        r"explique(?:-moi)? (?:pourquoi|comment)"
    ]
    
    # 3. Références à des entités spécifiques
    entity_patterns = [
        r"(?:le|la|les) (?:maire|forgeron|styliste|cuisinière)",
        r"(?:cette|cet|ces) (?:île|endroit|lieu)",
        r"(?:mon|ma|mes|notre|nos) (?:histoire|passé|famille)",
        r"(?:trésor|or|richesse|fortune)",
        r"(?:montgolfière|voyage|aventure)"
    ]
    
    # === ANALYSE ===
    
    score_rag = 0
    matched_indicators = []
    
    # Vérification des mots-clés
    for keyword in knowledge_keywords:
        if keyword in message_lower:
            score_rag += 2
            matched_indicators.append(f"keyword: {keyword}")
    
    # Vérification des patterns de questions
    for pattern in question_patterns:
        if re.search(pattern, message_lower):
            score_rag += 3
            matched_indicators.append(f"question_pattern: {pattern}")
    
    # Vérification des entités
    for pattern in entity_patterns:
        if re.search(pattern, message_lower):
            score_rag += 1
            matched_indicators.append(f"entity_pattern: {pattern}")
    
    # Bonus selon l'intention
    if intent in ["question", "request"]:
        score_rag += 1
        matched_indicators.append(f"intent: {intent}")
    
    # Bonus selon la complexité
    if state["complexity_level"] == "complex":
        score_rag += 1
        matched_indicators.append("complexity: complex")
    
    # === DÉCISION ===
    
    needs_rag = score_rag >= 2
    
    # Construction de la requête RAG si nécessaire
    query = None
    if needs_rag:
        query = _build_rag_query(message, intent, matched_indicators)
    
    return {
        "needs_rag": needs_rag,
        "score": score_rag,
        "matched_indicators": matched_indicators,
        "query": query,
        "reasoning": f"Score: {score_rag}, Seuil: 2, Indicators: {len(matched_indicators)}"
    }


def _build_rag_query(message: str, intent: str, indicators: List[str]) -> str:
    """Construit une requête optimisée pour le système RAG."""
    
    # Extraction des mots-clés importants
    important_words = []
    
    # Mots-clés selon l'intention
    if intent == "question":
        # Garde les mots interrogatifs et le sujet
        question_words = ["qui", "que", "quoi", "où", "quand", "comment", "pourquoi"]
        words = message.lower().split()
        for i, word in enumerate(words):
            if word in question_words and i + 1 < len(words):
                important_words.extend(words[i:i+3])  # Question + 2 mots suivants
    
    # Entités nommées simples
    entities = re.findall(r'\b(?:maire|forgeron|styliste|cuisinière|île|trésor|montgolfière)\b', 
                         message.lower())
    important_words.extend(entities)
    
    # Supprime les mots vides
    stop_words = {"le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais", "car"}
    important_words = [w for w in important_words if w not in stop_words and len(w) > 2]
    
    # Construction de la requête
    if important_words:
        query = " ".join(set(important_words))
    else:
        # Fallback : utilise le message original nettoyé
        query = re.sub(r'[^\w\s]', '', message).strip()
    
    return query[:100]  # Limite la longueur