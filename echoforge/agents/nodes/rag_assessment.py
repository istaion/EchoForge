"""Nœud d'évaluation du besoin de recherche RAG."""

import re
from typing import Dict, Any, List, Optional
from ..state.character_state import CharacterState
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()
@traceable
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
    character_name = state["character_name"]
    
    # Analyse par LLM du besoin RAG
    rag_analysis = _llm_rag_analysis(message, intent, character_name)
    
    # Mise à jour de l'état
    state["needs_rag_search"] = rag_analysis["needs_rag"]
    state["rag_query"] = rag_analysis["query"]
    
    # Debug info
    state["debug_info"]["rag_assessment"] = rag_analysis
    
    return state


def _llm_rag_analysis(message: str, intent: str, character_name: str) -> Dict[str, Any]:
    """Analyse par LLM du besoin de recherche RAG."""
    
    try:
        # Import du système LLM
        from echoforge.core import EchoForgeRAG
        from echoforge.utils.config import get_config
        
        # Récupération de la configuration et du LLM
        config = get_config()
        rag_system = EchoForgeRAG(
            data_path=str(config.data_path),
            vector_store_path=str(config.vector_store_path),
            embedding_model=config.embedding_model,
            llm_model=config.llm_model
        )
        
        # Construction du prompt d'évaluation
        evaluation_prompt = f"""Tu es un assistant qui détermine si une recherche de connaissances est nécessaire.

PERSONNAGE: {character_name}
MESSAGE UTILISATEUR: "{message}"
INTENTION: {intent}

Détermine si ce message nécessite une recherche dans les connaissances du personnage ou du monde.

CRITÈRES POUR RECHERCHE RAG:
- Questions sur l'histoire, le passé, les événements
- Demandes d'informations spécifiques sur le monde ou les relations
- Questions "pourquoi", "comment", "qui", "où", "quand"
- Références à des secrets, mystères, ou connaissances spécialisées
- Demandes d'explications détaillées

CRITÈRES POUR PAS DE RECHERCHE:
- Salutations simples ("bonjour", "salut")
- Réponses courtes ("oui", "non", "merci")
- Questions sur l'état actuel simple
- Conversations sociales basiques

Réponds EXACTEMENT dans ce format JSON:
{{
    "needs_rag": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explication courte",
    "query": "requête optimisée pour la recherche (si needs_rag=true, sinon null)"
}}

Exemple:
Message: "Raconte-moi l'histoire de l'île"
Réponse: {{"needs_rag": true, "confidence": 0.95, "reasoning": "Demande d'informations historiques spécifiques", "query": "histoire île événements passé"}}

Message: "Bonjour comment ça va"
Réponse: {{"needs_rag": false, "confidence": 0.9, "reasoning": "Salutation simple", "query": null}}

RÉPONSE:"""

        # Appel au LLM
        llm_response = rag_system.llm.invoke(evaluation_prompt)
        
        # Parse de la réponse JSON
        try:
            import json
            # Nettoie la réponse pour extraire le JSON
            response_clean = llm_response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:-3]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:-3]
            
            result = json.loads(response_clean)
            
            # Validation et structuration de la réponse
            return {
                "needs_rag": bool(result.get("needs_rag", False)),
                "query": result.get("query"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "Évaluation LLM"),
                "method": "llm_analysis",
                "raw_response": llm_response
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Erreur parsing réponse LLM: {e}")
            print(f"Réponse brute: {llm_response}")
            # Fallback sur analyse de mots-clés
            return _fallback_keyword_analysis(message, intent)
    
    except Exception as e:
        print(f"⚠️ Erreur lors de l'évaluation LLM RAG: {e}")
        # Fallback sur analyse basique
        return _fallback_keyword_analysis(message, intent)


def _fallback_keyword_analysis(message: str, intent: str) -> Dict[str, Any]:
    """Analyse de fallback basée sur des mots-clés."""
    
    message_lower = message.lower()
    
    # Mots-clés nécessitant une recherche de connaissances
    knowledge_keywords = [
        "histoire", "passé", "avant", "autrefois", "jadis",
        "secret", "mystère", "caché", "confidentiel",
        "relation", "ami", "ennemi", "famille",
        "événement", "incident", "accident", "guerre",
        "souvenir", "mémoire", "rappelle",
        "pourquoi", "comment", "raison", "cause",
        "qui est", "qu'est-ce que", "où se trouve",
        "origine", "création", "fondation",
        "tradition", "coutume", "rituel",
        "raconte", "explique"
    ]
    
    # Mots-clés simples (pas de RAG)
    simple_keywords = [
        "bonjour", "salut", "hey", "hello",
        "au revoir", "bye", "à bientôt",
        "merci", "de rien", "ok", "d'accord",
        "oui", "non", "peut-être",
        "ça va", "comment ça va"
    ]
    
    # Comptage des matches
    knowledge_score = sum(1 for keyword in knowledge_keywords if keyword in message_lower)
    simple_score = sum(1 for keyword in simple_keywords if keyword in message_lower)
    
    # Bonus pour certaines intentions
    if intent in ["question", "request"]:
        knowledge_score += 1
    elif intent in ["greeting", "farewell", "small_talk"]:
        simple_score += 2
    
    # Bonus pour la longueur du message
    word_count = len(message.split())
    if word_count > 10:
        knowledge_score += 1
    elif word_count <= 3:
        simple_score += 1
    
    # Décision
    needs_rag = knowledge_score > simple_score and knowledge_score > 0
    confidence = min(0.8, max(0.3, (knowledge_score - simple_score) / 5 + 0.5))
    
    # Construction de la requête si nécessaire
    query = None
    if needs_rag:
        # Extrait les mots importants pour la requête
        important_words = []
        for keyword in knowledge_keywords:
            if keyword in message_lower:
                important_words.append(keyword)
        
        # Ajoute d'autres mots significatifs
        words = message.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ["dans", "avec", "pour", "que", "qui"]:
                important_words.append(word.lower())
        
        query = " ".join(set(important_words))[:100]  # Limite la longueur
    
    return {
        "needs_rag": needs_rag,
        "query": query,
        "confidence": confidence,
        "reasoning": f"Analyse mots-clés: knowledge={knowledge_score}, simple={simple_score}",
        "method": "keyword_fallback",
        "knowledge_score": knowledge_score,
        "simple_score": simple_score
    }

# def assess_rag_need(state: CharacterState) -> CharacterState:
#     """
#     Évalue si une recherche RAG est nécessaire pour répondre à la requête.
    
#     Args:
#         state: État actuel du personnage
        
#     Returns:
#         État mis à jour avec l'évaluation RAG
#     """
#     state["processing_steps"].append("rag_assessment")
    
#     message = state["parsed_message"]
#     intent = state["message_intent"]
    
#     # Analyse détaillée du besoin RAG
#     rag_analysis = _detailed_rag_analysis(message, intent, state)
    
#     # Mise à jour de l'état
#     state["needs_rag_search"] = rag_analysis["needs_rag"]
#     state["rag_query"] = rag_analysis["query"]
    
#     # Debug info
#     state["debug_info"]["rag_assessment"] = rag_analysis
    
#     return state


# def _detailed_rag_analysis(message: str, intent: str, state: CharacterState) -> dict:
#     """Analyse détaillée du besoin de recherche RAG."""
#     message_lower = message.lower()
    
#     # === INDICATEURS DE BESOIN RAG ===
    
#     # 1. Mots-clés nécessitant une recherche de connaissances
#     knowledge_keywords = [
#         "histoire", "passé", "avant", "autrefois", "jadis",
#         "secret", "mystère", "caché", "confidentiel",
#         "relation", "ami", "ennemi", "famille",
#         "événement", "incident", "accident", "guerre",
#         "souvenir", "mémoire", "rappelle",
#         "pourquoi", "comment", "raison", "cause",
#         "qui est", "qu'est-ce que", "où se trouve",
#         "origine", "création", "fondation",
#         "tradition", "coutume", "rituel"
#     ]
    
#     # 2. Questions spécifiques
#     question_patterns = [
#         r"qui (?:est|était|a été)",
#         r"qu'est-ce que",
#         r"pourquoi (?:est-ce que|as-tu|avez-vous)",
#         r"comment (?:est-ce que|as-tu|avez-vous)",
#         r"où (?:est|se trouve|as-tu)",
#         r"quand (?:est-ce que|as-tu|avez-vous)",
#         r"raconte(?:-moi)? (?:l'histoire|le passé|les événements)",
#         r"explique(?:-moi)? (?:pourquoi|comment)"
#     ]
    
#     # 3. Références à des entités spécifiques
#     entity_patterns = [
#         r"(?:le|la|les) (?:maire|forgeron|styliste|cuisinière)",
#         r"(?:cette|cet|ces) (?:île|endroit|lieu)",
#         r"(?:mon|ma|mes|notre|nos) (?:histoire|passé|famille)",
#         r"(?:trésor|or|richesse|fortune)",
#         r"(?:montgolfière|voyage|aventure)"
#     ]
    
#     # === ANALYSE ===
    
#     score_rag = 0
#     matched_indicators = []
    
#     # Vérification des mots-clés
#     for keyword in knowledge_keywords:
#         if keyword in message_lower:
#             score_rag += 2
#             matched_indicators.append(f"keyword: {keyword}")
    
#     # Vérification des patterns de questions
#     for pattern in question_patterns:
#         if re.search(pattern, message_lower):
#             score_rag += 3
#             matched_indicators.append(f"question_pattern: {pattern}")
    
#     # Vérification des entités
#     for pattern in entity_patterns:
#         if re.search(pattern, message_lower):
#             score_rag += 1
#             matched_indicators.append(f"entity_pattern: {pattern}")
    
#     # Bonus selon l'intention
#     if intent in ["question", "request"]:
#         score_rag += 1
#         matched_indicators.append(f"intent: {intent}")
    
#     # Bonus selon la complexité
#     if state["complexity_level"] == "complex":
#         score_rag += 1
#         matched_indicators.append("complexity: complex")
    
#     # === DÉCISION ===
    
#     needs_rag = score_rag >= 2
    
#     # Construction de la requête RAG si nécessaire
#     query = None
#     if needs_rag:
#         query = _build_rag_query(message, intent, matched_indicators)
    
#     return {
#         "needs_rag": needs_rag,
#         "score": score_rag,
#         "matched_indicators": matched_indicators,
#         "query": query,
#         "reasoning": f"Score: {score_rag}, Seuil: 2, Indicators: {len(matched_indicators)}"
#     }

# @traceable
# def _build_rag_query(message: str, intent: str, indicators: List[str]) -> str:
#     """Construit une requête optimisée pour le système RAG."""
    
#     # Extraction des mots-clés importants
#     important_words = []
    
#     # Mots-clés selon l'intention
#     if intent == "question":
#         # Garde les mots interrogatifs et le sujet
#         question_words = ["qui", "que", "quoi", "où", "quand", "comment", "pourquoi"]
#         words = message.lower().split()
#         for i, word in enumerate(words):
#             if word in question_words and i + 1 < len(words):
#                 important_words.extend(words[i:i+3])  # Question + 2 mots suivants
    
#     # Entités nommées simples
#     entities = re.findall(r'\b(?:maire|forgeron|styliste|cuisinière|île|trésor|montgolfière)\b', 
#                          message.lower())
#     important_words.extend(entities)
    
#     # Supprime les mots vides
#     stop_words = {"le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais", "car"}
#     important_words = [w for w in important_words if w not in stop_words and len(w) > 2]
    
#     # Construction de la requête
#     if important_words:
#         query = " ".join(set(important_words))
#     else:
#         # Fallback : utilise le message original nettoyé
#         query = re.sub(r'[^\w\s]', '', message).strip()
    
#     return query[:100]  # Limite la longueur