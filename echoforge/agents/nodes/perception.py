"""Nœud de perception pour analyser l'input utilisateur."""

import time
import re
from typing import Dict, Any
from ..state.character_state import CharacterState, ComplexityLevel
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()
@traceable
def perceive_input(state: CharacterState) -> CharacterState:
    """
    Nœud de perception - analyse l'input utilisateur et initialise l'état.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec l'analyse de l'input
    """
    # Initialisation du timing
    state["processing_start_time"] = time.time()
    state["processing_steps"] = ["perception"]
    state["debug_info"] = {}
    
    # Nettoyage et parsing basique du message
    user_message = state["user_message"].strip()
    state["parsed_message"] = user_message
    
    # Analyse de l'intention basique
    intent = _analyze_message_intent(user_message)
    state["message_intent"] = intent
    
    # Évaluation de la complexité initiale
    complexity = _assess_initial_complexity(user_message, intent)
    state["complexity_level"] = complexity
    
    # Initialisation des flags
    state["needs_rag_search"] = False
    state["rag_results"] = []
    state["relevant_knowledge"] = []
    state["planned_actions"] = []
    state["triggered_events"] = []
    state["game_state_changes"] = {}
    
    # Debug info
    state["debug_info"]["perception"] = {
        "original_message": user_message,
        "detected_intent": intent,
        "initial_complexity": complexity
    }
    
    return state


def _analyze_message_intent(message: str) -> str:
    """Analyse l'intention du message utilisateur."""
    message_lower = message.lower()
    
    # Patterns d'intention
    intent_patterns = {
        "greeting": ["bonjour", "salut", "hey", "hello", "bonsoir"],
        "farewell": ["au revoir", "bye", "à bientôt", "adieu"],
        "question": ["?", "pourquoi", "comment", "quoi", "qui", "où", "quand"],
        "request": ["peux-tu", "pourrais-tu", "donne-moi", "montre-moi"],
        "transaction": ["acheter", "vendre", "échanger", "troquer", "prix"],
        "emotional": ["triste", "heureux", "en colère", "content", "désolé"],
        "game_action": ["prendre", "utiliser", "aller", "parler", "donner"],
        "small_talk": ["comment ça va", "quoi de neuf", "ça va", "météo"]
    }
    
    # Cherche le premier pattern qui match
    for intent, patterns in intent_patterns.items():
        if any(pattern in message_lower for pattern in patterns):
            return intent
    
    return "general"


def _assess_initial_complexity(message: str, intent: str) -> ComplexityLevel:
    """Évalue la complexité initiale du message."""
    message_lower = message.lower()
    
    # Patterns simples (réponses cachées/automatiques)
    simple_patterns = [
        "bonjour", "salut", "merci", "de rien", "oui", "non", 
        "d'accord", "ok", "ça va", "hello"
    ]
    
    # Patterns complexes (nécessitent réflexion/RAG)
    complex_patterns = [
        "raconte", "explique", "histoire", "pourquoi", "comment",
        "secret", "relation", "passé", "souvenir", "événement"
    ]
    
    # Vérification longueur
    word_count = len(message.split())
    
    # Classification
    if any(pattern in message_lower for pattern in simple_patterns) and word_count <= 3:
        return ComplexityLevel.SIMPLE
    
    if any(pattern in message_lower for pattern in complex_patterns) or word_count > 15:
        return ComplexityLevel.COMPLEX
    
    # Selon l'intention
    if intent in ["greeting", "farewell", "small_talk"]:
        return ComplexityLevel.SIMPLE
    elif intent in ["question", "request", "emotional"]:
        return ComplexityLevel.COMPLEX
    
    return ComplexityLevel.MEDIUM