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
    state["rag_query"] = [rag_analysis["query"]]
    
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
            llm_provider=config.llm_provider,
            llm_model=config.llm_model
        )
        
        # Construction du prompt d'évaluation
        evaluation_prompt = f"""Tu es un assistant qui détermine si une recherche de connaissances est nécessaire.

PERSONNAGE: {character_name}
MESSAGE UTILISATEUR: "{message}"
INTENTION: {intent}

Détermine si ce message nécessite une recherche dans les connaissances du personnage ou du monde.

CRITÈRES POUR RECHERCHE RAG:
- Questions sur l’histoire, le passé ou l’enfance du personnage
- Demandes sur les événements passés, souvenirs ou motivations
- Recherches sur le monde, les lieux, les autres personnages
- Questions "pourquoi", "comment", "qui", "où", "quand"
- Références à des secrets, mystères, ou connaissances spécialisées
- Demandes d’explications ou d’histoires détaillées

CRITÈRES POUR PAS DE RECHERCHE:
- Salutations simples ("bonjour", "salut")
- Réponses brèves ("oui", "non", "merci")
- Réactions sociales immédiates sans contenu narratif
- Questions très génériques ou déjà connues du personnage sans contexte

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

Message: "Parle-moi de ton passé"
Réponse: {{"needs_rag": true, "confidence": 0.92, "reasoning": "Demande directe sur le passé du personnage", "query": "passé enfance souvenirs"}}

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
            query = result.get("query")
            if isinstance(query, str) and query.strip().lower() == "null":
                query = None
            
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

def validate_rag_results(state: CharacterState) -> CharacterState:
    """
    Vérifie si les résultats RAG sont suffisamment pertinents pour continuer.
    Marque si une nouvelle recherche est nécessaire.
    """
    state["processing_steps"].append("rag_validation")
    if len(state["rag_query"]) >= 3:  # 1 original + 2 reformulations
        state["needs_rag_retry"] = False
        state["rag_retry_reason"] = "Nombre maximum de tentatives RAG atteint"
        state["relevant_knowledge"] = _select_best_knowledge_for_generation(state)
    else:
        results = state.get("rag_results", [])
        top_score = results[0]["relevance"] if results else 0.0
        min_acceptable_score = 0.6  # ← configurable

        if top_score < min_acceptable_score:
            # Pas assez d'infos utiles : nouvelle tentative ?
            state["debug_info"]["rag_validation"] = {
                "status": "insufficient_information",
                "top_score": top_score
            }
            state["needs_rag_retry"] = True
            state["rag_retry_reason"] = (
                "Les connaissances récupérées sont trop peu pertinentes "
                f"(score max = {top_score:.2f} < seuil {min_acceptable_score})."
            )
        else:
            state["debug_info"]["rag_validation"] = {
                "status": "ok",
                "top_score": top_score
            }
            state["needs_rag_retry"] = False
            state["rag_retry_reason"] = None

    return state

def _select_best_knowledge_for_generation(state: CharacterState, top_k: int = 5) -> List[str]:
    """
    Sélectionne les contenus RAG les plus pertinents pour la génération finale.
    """
    all_results = state.get("rag_results", [])
    selected = sorted(all_results, key=lambda x: x["relevance"], reverse=True)[:top_k]
    return [r["content"] for r in selected]

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
        
        query = " ".join(set(important_words))[:200]  # Limite la longueur
    
    return {
        "needs_rag": needs_rag,
        "query": query,
        "confidence": confidence,
        "reasoning": f"Analyse mots-clés: knowledge={knowledge_score}, simple={simple_score}",
        "method": "keyword_fallback",
        "knowledge_score": knowledge_score,
        "simple_score": simple_score
    }
