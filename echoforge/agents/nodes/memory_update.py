"""Nœud de mise à jour de la mémoire du personnage."""

import time
from typing import Dict, Any
from ..state.character_state import CharacterState
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()
@traceable
def update_character_memory(state: CharacterState) -> CharacterState:
    """
    Met à jour la mémoire du personnage avec l'interaction actuelle.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la mémoire actualisée
    """
    state["processing_steps"].append("memory_update")
    
    # Création de l'entrée mémoire
    memory_entry = _create_memory_entry(state)
    
    # Mise à jour de l'historique de conversation
    state["conversation_history"].append({
        "user": state["user_message"],
        "assistant": state["response"],
        "timestamp": time.time(),
        "metadata": memory_entry["metadata"]
    })
    
    # Limitation de l'historique (garde les 10 derniers échanges)
    if len(state["conversation_history"]) > 10:
        state["conversation_history"] = state["conversation_history"][-10:]
    
    # Debug info
    state["debug_info"]["memory_update"] = {
        "entry_created": True,
        "conversation_length": len(state["conversation_history"]),
        "memory_importance": memory_entry["importance"],
        "emotional_impact": memory_entry["emotional_impact"]
    }
    
    return state


def _create_memory_entry(state: CharacterState) -> Dict[str, Any]:
    """Crée une entrée de mémoire pour l'interaction actuelle."""
    
    # Calcul de l'importance de l'interaction
    importance = _calculate_importance(state)
    
    # Calcul de l'impact émotionnel
    emotional_impact = _calculate_emotional_impact(state)
    
    # Extraction des concepts clés
    key_concepts = _extract_key_concepts(state)
    
    memory_entry = {
        "timestamp": time.time(),
        "user_message": state["user_message"],
        "response": state["response"],
        "importance": importance,
        "emotional_impact": emotional_impact,
        "key_concepts": key_concepts,
        "metadata": {
            "intent": state["message_intent"],
            "complexity": state["complexity_level"],
            "used_rag": bool(state["rag_results"]),
            "processing_steps": state["processing_steps"],
            "character_emotion": state["current_emotion"]
        }
    }
    
    return memory_entry


def _calculate_importance(state: CharacterState) -> float:
    """Calcule l'importance de l'interaction (0.0 à 1.0)."""
    
    importance = 0.3  # Base
    
    # Bonus selon la complexité
    if state["complexity_level"] == "complex":
        importance += 0.3
    elif state["complexity_level"] == "medium":
        importance += 0.1
    
    # Bonus si RAG utilisé (information importante partagée)
    if state["rag_results"]:
        importance += 0.2
    
    # Bonus selon l'intention
    high_importance_intents = ["question", "request", "emotional", "transaction"]
    if state["message_intent"] in high_importance_intents:
        importance += 0.2
    
    # Bonus selon la longueur du message
    word_count = len(state["user_message"].split())
    if word_count > 10:
        importance += 0.1
    
    # Bonus si actions planifiées
    if state["planned_actions"]:
        importance += 0.2
    
    return min(importance, 1.0)


def _calculate_emotional_impact(state: CharacterState) -> float:
    """Calcule l'impact émotionnel de l'interaction (-1.0 à 1.0)."""
    
    emotional_impact = 0.0
    
    message = state["user_message"].lower()
    
    # Mots positifs
    positive_words = [
        "merci", "génial", "parfait", "excellent", "super", "formidable",
        "content", "heureux", "joie", "plaisir", "aime", "adore"
    ]
    
    # Mots négatifs
    negative_words = [
        "désolé", "triste", "énervé", "colère", "déteste", "horrible",
        "nul", "mauvais", "déçu", "frustré", "problème", "ennui"
    ]
    
    # Calcul du score
    for word in positive_words:
        if word in message:
            emotional_impact += 0.2
    
    for word in negative_words:
        if word in message:
            emotional_impact -= 0.2
    
    # Ajustement selon l'intention
    if state["message_intent"] == "emotional":
        emotional_impact *= 1.5  # Amplifie l'impact émotionnel
    
    return max(-1.0, min(1.0, emotional_impact))


def _extract_key_concepts(state: CharacterState) -> list:
    """Extrait les concepts clés de l'interaction."""
    
    message = state["user_message"].lower()
    concepts = []
    
    # Entités importantes
    entities = [
        "trésor", "or", "cookies", "tissu", "montgolfière",
        "île", "maire", "forgeron", "styliste", "cuisinière",
        "réparation", "échange", "commerce", "histoire", "secret"
    ]
    
    for entity in entities:
        if entity in message:
            concepts.append(entity)
    
    # Concepts des résultats RAG
    if state["rag_results"]:
        for result in state["rag_results"]:
            if result["relevance"] > 0.7:
                # Extraction simple de mots-clés du contenu RAG
                content_words = result["content"].lower().split()
                important_words = [w for w in content_words if len(w) > 4 and w in entities]
                concepts.extend(important_words)
    
    # Actions mentionnées
    action_words = [
        "donner", "prendre", "acheter", "vendre", "réparer", "créer",
        "cuisiner", "forger", "coudre", "échanger"
    ]
    
    for action in action_words:
        if action in message:
            concepts.append(f"action_{action}")
    
    return list(set(concepts))  # Supprime les doublons


def finalize_interaction(state: CharacterState) -> CharacterState:
    """
    Finalise l'interaction et prépare l'état pour la prochaine.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État finalisé
    """
    state["processing_steps"].append("finalization")
    
    # Calcul du temps de traitement total
    processing_time = time.time() - state["processing_start_time"]
    
    # Mise à jour des métadonnées finales
    state["debug_info"]["final_stats"] = {
        "total_processing_time": processing_time,
        "steps_count": len(state["processing_steps"]),
        "rag_used": bool(state["rag_results"]),
        "response_length": len(state["response"]),
        "complexity_route": state["complexity_level"]
    }
    
    # Nettoyage des données temporaires si nécessaire
    # (garde les informations importantes pour la prochaine interaction)
    
    return state