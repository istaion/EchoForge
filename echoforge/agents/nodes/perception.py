import time
import re
from typing import Dict, Any
from ..state.character_state import CharacterState
from langsmith import traceable
from echoforge.utils.config import get_config
from echoforge.core.llm_providers import LLMManager
import json

config = get_config()
def interpret_player_input_node(llm_manager: LLMManager):
    """
    Nœud d'interprétation de l'input du joueur - analyse les intentions
    """
    def fn(state: CharacterState) -> CharacterState:
        user_msg = state["user_message"]
        trigger_defs = state['character_data']['triggers']['input']
        keys = list(trigger_defs.keys())
        example_json = "{\n" + ",\n".join([f'  \"{k}\": 0.0' for k in keys]) + "\n}"

        # Décris les triggers en texte (pas en JSON !)
        trigger_descriptions = "\n".join([f"- {k} : {v['trigger']}" for k, v in trigger_defs.items()])

        
        # Prompt pour scorer les probabilités de triggers
        prompt = f"""
Tu es un détecteur d’intentions pour un personnage de jeu de rôle.

Le joueur vient d’écrire :
\"{user_msg}\"

Voici les intentions possibles :
{trigger_descriptions}

Pour chaque intention, donne une **probabilité** entre 0.0 (pas du tout présent) et 1.0 (certain) que cette intention soit exprimée dans le message du joueur.

Répond uniquement avec un objet JSON au format :
{example_json}
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
    
    return fn

def decide_intent_node():
    """
    Nœud de décision d'intention - détermine les triggers activés
    """
    def fn(state: CharacterState) -> CharacterState:
        input_trigger_config = state['character_data']['triggers']["input"]
        input_trigger_scores = state.get("input_trigger_probs", {})
        
        activated = []
        refused = []
        best_trigger = None
        best_score = 0.0
        
        for trigger_name, score in input_trigger_scores.items():
            threshold = input_trigger_config.get(trigger_name, {}).get("threshold", 0.5)
            if score >= threshold:
                activated.append(trigger_name)
                if score > best_score:
                    best_score = score
                    best_trigger = trigger_name
            else:
                refused.append(trigger_name)
        
        # Mise à jour du state
        state["activated_input_triggers"] = activated
        state["message_intent"] = best_trigger if best_score > 0.5 else None
        state["processing_steps"].append("decide_intent")
        
        return state
    
    return fn


def interpret_character_output(llm_manager: LLMManager):
    """
    Analyse la réponse générée par le personnage et détecte les déclencheurs de sortie.
    Retourne un dict avec probabilité et éventuellement une valeur associée.
    """
    def fn(state: CharacterState) -> CharacterState:
        response_text = state.get("response", "")
        output_triggers = state["character_data"].get("triggers", {}).get("output", {})

        # Construction de la description lisible des triggers
        trigger_descriptions = []
        for name, info in output_triggers.items():
            line = f"- {name}: {info.get('trigger', '')}"
            if "value_key" in info:
                line += f" (valeur attendue: {info['value_key']})"
            trigger_descriptions.append(line)

        # Construction d’un exemple clair pour le LLM
        example_object = "{\n" + ",\n".join([
            f'  "{name}": {{"prob": 0.0{" , \"value\": 0" if "value_key" in info else ""}}}'
            for name, info in output_triggers.items()
        ]) + "\n}"

        # Prompt LLM
        prompt = f"""
Tu es un analyseur narratif. Voici une phrase prononcée par un personnage de jeu de rôle :
\"\"\"{response_text}\"\"\"

Tu dois détecter s’il a exprimé une des conséquences narratives suivantes :
{chr(10).join(trigger_descriptions)}

Pour chaque déclencheur :
- Donne une probabilité (`prob`) entre 0.0 et 1.0
- Si le déclencheur implique une valeur (ex : nombre d’or donné), indique `value`

Répond uniquement avec un objet JSON de ce format :
{example_object}
"""

        try:
            raw_output = llm_manager.invoke(prompt)
            trigger_outputs = json.loads(raw_output)
        except Exception as e:
            state["debug_info"]["output_trigger_parser_error"] = str(e)
            trigger_outputs = {k: {"prob": 0.0} for k in output_triggers}

        # Validation et fallback : s'assurer que le format est bon
        for key in output_triggers:
            if key not in trigger_outputs:
                trigger_outputs[key] = {"prob": 0.0}
            else:
                if "prob" not in trigger_outputs[key]:
                    trigger_outputs[key]["prob"] = 0.0
                if "value" in output_triggers[key] and "value" not in trigger_outputs[key]:
                    # valeur attendue mais non précisée → défaut ?
                    trigger_outputs[key]["value"] = output_triggers[key].get("value_default", 0)

        # Enregistrement dans le state
        state["output_trigger_probs"] = trigger_outputs
        state["processing_steps"].append("interpret_character_output")

        return state
    return fn

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
    
    # # Évaluation de la complexité initiale
    # complexity = _assess_initial_complexity(user_message, intent)
    # state["complexity_level"] = complexity
    
    # Initialisation des flags
    state["needs_rag_search"] = False
    state["rag_results"] = []
    state["relevant_knowledge"] = []
    state["game_state_changes"] = {}
    
    # Debug info
    state["debug_info"]["perception"] = {
        "original_message": user_message,
        "detected_intent": intent
    }
    # ,
    #     "initial_complexity": complexity
    
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


# def _assess_initial_complexity(message: str, intent: str) -> ComplexityLevel:
#     """Évalue la complexité initiale du message."""
#     message_lower = message.lower()
    
#     # Patterns simples (réponses cachées/automatiques)
#     simple_patterns = [
#         "bonjour", "salut", "merci", "de rien", "oui", "non", 
#         "d'accord", "ok", "ça va", "hello"
#     ]
    
#     # Patterns complexes (nécessitent réflexion/RAG)
#     complex_patterns = [
#         "raconte", "explique", "histoire", "pourquoi", "comment",
#         "secret", "relation", "passé", "souvenir", "événement"
#     ]
    
#     # Vérification longueur
#     word_count = len(message.split())
    
#     # Classification
#     if any(pattern in message_lower for pattern in simple_patterns) and word_count <= 3:
#         return ComplexityLevel.SIMPLE
    
#     if word_count > 15:
#         return ComplexityLevel.COMPLEX
    
#     return ComplexityLevel.MEDIUM