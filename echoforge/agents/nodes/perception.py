import time
import re
from typing import Dict, Any
from ..state.character_state import CharacterState, ComplexityLevel
from langsmith import traceable
from echoforge.utils.config import get_config
from langgraph.graph import StateNode
from core.llm_providers import LLMManager

config = get_config()
@traceable
def interpret_player_input_node(llm_manager: LLMManager) -> StateNode:
    def fn(state: CharacterState) -> CharacterState:
        user_msg = state["user_message"]
        
        # Prompt pour scorer les probabilités de triggers
        prompt = f"""
Tu es un détecteur d’intention dans un jeu de rôle narratif. Le message du joueur est :

"{user_msg}"

Voici la liste des déclencheurs possibles :
{state['input_trigger_probs']}

Pour chaque déclencheur, donne une probabilité entre 0.0 et 1.0 qu’il soit présent dans ce message, au format JSON :
{{
  "bye": ...,
  "ask_for_money": ...
}}

Répond uniquement avec l'objet JSON.
"""
        result = llm_manager.invoke(prompt)

        try:
            trigger_probs = json.loads(result)
        except json.JSONDecodeError:
            trigger_probs = {"bye": 0.0, "ask_for_money": 0.0}
            state["debug_info"]["interpret_error"] = "LLM returned invalid JSON"

        # Choix du message_intent le plus probable
        best_trigger = max(trigger_probs.items(), key=lambda x: x[1])[0]
        best_score = trigger_probs[best_trigger]

        # Mise à jour du state
        state["parsed_message"] = user_msg
        state["message_intent"] = best_trigger if best_score > 0.5 else None
        state["input_trigger_probs"] = trigger_probs
        state["processing_steps"].append("interpret_player_input")

        return state

    return StateNode(fn=fn)

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
    
    if word_count > 15:
        return ComplexityLevel.COMPLEX
    
    return ComplexityLevel.MEDIUM