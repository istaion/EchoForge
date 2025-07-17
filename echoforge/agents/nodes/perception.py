import time
import re
from typing import Dict, Any
from functools import partial
from ..state.character_state import CharacterState
from langchain.tools import tool
from langchain_core.tools import Tool
from langsmith import traceable
from echoforge.utils.config import get_config
from echoforge.core.llm_providers import LLMManager
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

def make_get_player_state_tool(player_data: dict):
    return Tool(
        name="get_player_state_flat",
        description="Retourne les données du joueur. IMPORTANT: Cette fonction ne prend AUCUN argument. Appelle-la sans paramètres.",
        func=lambda: player_data,
    )

def make_get_character_state_tool(character_data: dict):
    return Tool(
        name="get_character_state",
        description="Retourne les données du personnage. IMPORTANT: Cette fonction ne prend AUCUN argument. Appelle-la sans paramètres.",
        func=lambda: character_data,
    )

@traceable
def interpret_triggers_input_node(llm_manager: LLMManager):
    def fn(state: CharacterState) -> CharacterState:
        triggers = state["character_data"]["triggers"]["input"]
        player_msg = state["user_message"]
        player_data = state["player_data"]
        character_data = {k: v for k, v in state["character_data"].items() if k != "triggers"}

        # Crée les tools
        tools = [
            make_get_player_state_tool(player_data),
            make_get_character_state_tool(character_data),
        ]
        
        tool_bound_llm = llm_manager.bind_tools(tools)

        system_prompt = """Tu es un détecteur d'intentions. Tu scores les triggers, puis vérifies les conditions si nécessaires.

PROCESSUS:
1. Analyse d'abord le message du joueur
2. SI tu as besoin de données spécifiques pour vérifier les conditions, utilise les tools
3. Évalue ensuite chaque trigger avec probabilité (0.0 à 1.0)
4. Vérifie les conditions pour déterminer les triggers activés/refusés
5. Retourne TOUJOURS le JSON final avec les résultats

TOOLS DISPONIBLES:
- get_player_state_flat(): données du joueur (or, cookies, alcool, etc.)
- get_character_state(): données du personnage (relation, état, etc.)

IMPORTANT: Les tools ne prennent AUCUN argument. Appelle-les sans paramètres."""

        user_prompt = f"""Message du joueur : "{player_msg}"

Triggers disponibles :
{json.dumps(triggers, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Analyse le message pour détecter les intentions
2. Si nécessaire pour vérifier les conditions, utilise get_player_state_flat() et/ou get_character_state()
3. Évalue chaque trigger (probabilité 0.0 à 1.0)
4. Vérifie les conditions (ex: "relation > 5", "alcool in possession")
5. Retourne le JSON final avec les résultats

FORMAT DE RÉPONSE OBLIGATOIRE:
{{
    "input_trigger_probs": {{"bye": 0.0, "ask_for_money": 0.0, "give_alcool": 0.0, "ask_for_treasure": 0.0}},
    "activated_input_triggers": ["trigger_name"],
    "refused_input_triggers": [{{"trigger": "trigger_name", "reason_refused": "raison"}}]
}}"""

        try:
            # Invoque le LLM avec tools
            messages = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]
            
            response = tool_bound_llm.invoke(messages)
            
            # 🔧 SOLUTION: Gestion complète des tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Tool calls détectés: {len(response.tool_calls)}")
                
                # Exécute les tool calls
                messages_with_tools = list(messages)
                messages_with_tools.append(response)
                
                for tool_call in response.tool_calls:
                    print(f"🔧 Exécution du tool: {tool_call['name']}")
                    
                    # Trouve et exécute le tool
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_call['name']:
                            try:
                                tool_result = tool.func()
                                print(f"✅ Tool {tool_call['name']} exécuté avec succès")
                                break
                            except Exception as e:
                                print(f"❌ Erreur tool {tool_call['name']}: {e}")
                                tool_result = f"Erreur: {e}"
                    
                    # Ajoute le résultat du tool aux messages
                    tool_message = ToolMessage(
                        content=str(tool_result) if tool_result else "Aucun résultat",
                        tool_call_id=tool_call['id']
                    )
                    messages_with_tools.append(tool_message)
                
                # Demande au LLM de continuer avec les résultats des tools
                continuation_prompt = """Maintenant que tu as les données nécessaires, analyse les triggers et retourne le JSON final.

Rappel du format obligatoire:
{
    "input_trigger_probs": {"bye": 0.0, "ask_for_money": 0.0, "give_alcool": 0.0, "ask_for_treasure": 0.0},
    "activated_input_triggers": ["trigger_name"],
    "refused_input_triggers": [{"trigger": "trigger_name", "reason_refused": "raison"}]
}"""
                
                messages_with_tools.append(("user", continuation_prompt))
                
                # Nouvelle invocation sans tools pour obtenir le JSON final
                final_response = llm_manager.get_llm().invoke(messages_with_tools)
                
                # Extrait le JSON de la réponse finale
                if hasattr(final_response, 'content'):
                    final_content = final_response.content
                else:
                    final_content = str(final_response)
                
                print(f"🔍 Réponse finale après tools: {final_content}")
                
            else:
                # Pas de tool calls, utilise la réponse directe
                if hasattr(response, 'content'):
                    final_content = response.content
                else:
                    final_content = str(response)
                
                print(f"🔍 Réponse directe sans tools: {final_content}")
            
            # Parse le JSON final
            final_json = extract_json_block(final_content)
            
            if not final_json:
                raise ValueError("Aucun JSON trouvé dans la réponse finale")
            
            parsed = json.loads(final_json)
            print(f"✅ Triggers parsés avec tools: {parsed}")
            
        except Exception as e:
            print(f"❌ Erreur avec la version tools: {e}")
            
            # Fallback vers analyse simple
            print("🔄 Basculement vers analyse simple...")

        # Validation et nettoyage des résultats
        parsed = validate_and_clean_trigger_results(parsed, triggers)

        # Injection dans le state
        state["input_trigger_probs"] = parsed.get("input_trigger_probs", {})
        state["activated_input_triggers"] = parsed.get("activated_input_triggers", [])
        state["refused_input_triggers"] = parsed.get("refused_input_triggers", [])
        state["processing_steps"].append("interpret_triggers_with_tools")

        return state

    return fn


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
        result = extract_json_block(str(result))
        
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
            clean_output = extract_json_block(str(raw_output))
            trigger_outputs = json.loads(clean_output)
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


def validate_and_clean_trigger_results(parsed: dict, triggers: dict) -> dict:
    """Valide et nettoie les résultats des triggers"""
    
    # Assure que toutes les clés existent
    if "input_trigger_probs" not in parsed:
        parsed["input_trigger_probs"] = {}
    if "activated_input_triggers" not in parsed:
        parsed["activated_input_triggers"] = []
    if "refused_input_triggers" not in parsed:
        parsed["refused_input_triggers"] = []
    
    # Assure que tous les triggers sont présents dans les probabilités
    for trigger_name in triggers.keys():
        if trigger_name not in parsed["input_trigger_probs"]:
            parsed["input_trigger_probs"][trigger_name] = 0.0
    
    # Nettoie les triggers inexistants
    valid_triggers = set(triggers.keys())
    parsed["activated_input_triggers"] = [
        t for t in parsed["activated_input_triggers"] 
        if t in valid_triggers
    ]
    
    return parsed

def extract_json_block(text: str) -> str:
    """
    Extrait le contenu JSON entre la première accolade ouvrante { et
    la dernière accolade fermante }, même si le texte a des préfixes ou suffixes.
    """
    try:
        text = str(text)
        
        # Nettoyer le texte
        text = text.strip()
        
        # Chercher le JSON avec différentes approches
        # 1. Entre ``` json et ```
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1)
        
        # 2. Match simple entre { et }
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # 3. Si le texte commence déjà par {
        if text.startswith('{'):
            return text
            
        return ""
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction JSON: {e}")
        return ""