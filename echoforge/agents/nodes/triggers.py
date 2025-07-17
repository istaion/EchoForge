"""
Agent ReAct pour l'analyse des triggers d'input avec vérification conditionnelle.
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool, BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field
import json
import re
from echoforge.agents.state.character_state import CharacterState
from echoforge.core.llm_providers import LLMManager


class TriggerAnalysisResult(BaseModel):
    """Résultat de l'analyse des triggers"""
    input_trigger_probs: Dict[str, float] = Field(description="Probabilités pour chaque trigger")
    activated_input_triggers: List[str] = Field(description="Triggers activés")
    refused_input_triggers: List[Dict[str, str]] = Field(description="Triggers refusés avec raisons")


class GetPlayerDataTool(BaseTool):
    """Tool pour récupérer les données du joueur"""
    name: str = "get_player_data"
    description: str = "Récupère les données complètes du joueur. Utilise ce tool pour vérifier les conditions liées au joueur (inventaire, stats, etc)."
    player_data: Dict[str, Any]

    def _run(self, query: str = "") -> str:
        return json.dumps(self.player_data, indent=2)

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class GetCharacterDataTool(BaseTool):
    name: str = "get_character_data"
    description: str = "Récupère les données complètes du personnage. Utilise ce tool pour vérifier les conditions liées au personnage (relation, état, etc)."
    character_data: Dict[str, Any]

    def _run(self, query: str = "") -> str:
        data_without_triggers = {k: v for k, v in self.character_data.items() if k != "triggers"}
        return json.dumps(data_without_triggers, indent=2)

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class EvaluateConditionTool(BaseTool):
    """Tool pour évaluer une condition"""
    name: str = "evaluate_condition"
    description: str = "Évalue une condition logique. Passe la condition comme string et les données nécessaires."
    
    def _run(self, condition: str, player_data: Dict = None, character_data: Dict = None) -> bool:
        """Execute the tool"""
        try:
            # Créer un contexte d'évaluation sécurisé
            context = {}
            
            if player_data:
                # Aplatir les données du joueur pour un accès facile
                for key, value in player_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            context[f"{key}_{sub_key}"] = sub_value
                    else:
                        context[key] = value
            
            if character_data:
                # Ajouter les données du personnage
                for key, value in character_data.items():
                    if key != "triggers":  # Ignorer les triggers
                        context[key] = value
            
            # Remplacer les variables dans la condition
            evaluated_condition = condition
            for var, val in context.items():
                if isinstance(val, str):
                    evaluated_condition = evaluated_condition.replace(var, f"'{val}'")
                else:
                    evaluated_condition = evaluated_condition.replace(var, str(val))
            
            # Évaluation sécurisée
            return eval(evaluated_condition, {"__builtins__": {}}, context)
        except Exception as e:
            return False
    
    async def _arun(self, condition: str, player_data: Dict = None, character_data: Dict = None) -> bool:
        return self._run(condition, player_data, character_data)


def create_trigger_analysis_prompt() -> PromptTemplate:
    """Crée le prompt pour l'agent d'analyse des triggers"""
    
    template = """Tu es un analyseur d'intentions expert. Tu dois analyser le message utilisateur et déterminer quels triggers sont activés.

Voici les triggers disponibles :
{triggers_json}

Message utilisateur : "{user_message}"

INSTRUCTIONS CRITIQUES :
1. D'abord, dans ta tête (dans le champ Thought:), analyse le message et attribue une probabilité (entre 0.0 et 1.0) à chaque trigger. **Ne fais aucune action ici.**
2. Si la probabilité d’un trigger dépasse son threshold :
   - Si le trigger n’a pas de conditions, considère-le comme activé
   - Sinon, utilise les tools pour vérifier si ses conditions sont remplies
3. Les triggers refusés doivent avoir une raison claire

Tu as accès aux tools suivants :
{tools}

Utilise le format suivant :

Question: la question d'entrée à laquelle tu dois répondre
Thought: tu dois toujours réfléchir à ce que tu dois faire
Action: l'action à prendre, doit être l'une de [{tool_names}]
Action Input: l'entrée de l'action
Observation: le résultat de l'action
... (cette séquence Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale doit être un JSON valide au format suivant :

{{
    "input_trigger_probs": {{"trigger_name": probability, ...}},
    "activated_input_triggers": ["trigger1", "trigger2", ...],
    "refused_input_triggers": [
        {{"trigger": "trigger_name", "reason_refused": "raison claire"}},
        ...
    ]
}}

Commence !

Question: Quels triggers sont activés pour ce message ?
{agent_scratchpad}"""
    
    return PromptTemplate(
        template=template,
        input_variables=["triggers_json", "user_message", "tools", "tool_names", "agent_scratchpad"]
    )


class TriggerAnalysisAgent:
    """Agent pour analyser les triggers d'input"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_llm()
    
    def analyze_triggers(
        self,
        user_message: str,
        triggers: Dict[str, Any],
        player_data: Dict[str, Any],
        character_data: Dict[str, Any]
    ) -> TriggerAnalysisResult:
        """
        Analyse les triggers pour un message utilisateur
        
        Args:
            user_message: Message de l'utilisateur
            triggers: Dictionnaire des triggers d'input
            player_data: Données du joueur
            character_data: Données du personnage
            
        Returns:
            TriggerAnalysisResult avec les probabilités et triggers activés/refusés
        """
        
        # Créer les tools
        tools = [
            GetPlayerDataTool(player_data=player_data),
            GetCharacterDataTool(character_data=character_data),
            EvaluateConditionTool()
        ]
        
        # Créer le prompt
        prompt = create_trigger_analysis_prompt()
        
        # Créer l'agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Créer l'executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        # Préparer les inputs
        triggers_json = json.dumps(triggers, indent=2, ensure_ascii=False)
        
        # Exécuter l'agent
        try:
            result = agent_executor.invoke({
                "triggers_json": triggers_json,
                "user_message": user_message
            })
            
            # Parser le résultat
            final_answer = result.get("output", "{}")
            
            # Extraire le JSON de la réponse
            json_match = re.search(r'\{[^{}]*\}', final_answer, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                return TriggerAnalysisResult(**parsed_result)
            else:
                # Fallback si pas de JSON trouvé
                return self._fallback_analysis(user_message, triggers)
                
        except Exception as e:
            print(f"Erreur dans l'agent d'analyse des triggers: {e}")
            return self._fallback_analysis(user_message, triggers)
    
    def _fallback_analysis(self, user_message: str, triggers: Dict[str, Any]) -> TriggerAnalysisResult:
        """Analyse de fallback simple basée sur des mots-clés"""
        
        result = TriggerAnalysisResult(
            input_trigger_probs={},
            activated_input_triggers=[],
            refused_input_triggers=[]
        )
        
        message_lower = user_message.lower()
        
        # Analyse simple par mots-clés
        for trigger_name, trigger_data in triggers.items():
            trigger_desc = trigger_data.get("trigger", "").lower()
            
            # Calcul simple de probabilité basé sur les mots-clés
            probability = 0.0
            
            # Recherche de mots-clés communs
            keywords = self._extract_keywords(trigger_desc)
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            
            if matches > 0:
                probability = min(1.0, matches * 0.3)
            
            result.input_trigger_probs[trigger_name] = probability
            
            # Vérifier le threshold
            threshold = trigger_data.get("threshold", 0.5)
            if probability < threshold:
                result.refused_input_triggers.append({
                    "trigger": trigger_name,
                    "reason_refused": "Le seuil n'est pas atteint"
                })
            else:
                # Dans le fallback, on active sans vérifier les conditions
                result.activated_input_triggers.append(trigger_name)
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrait des mots-clés simples d'un texte"""
        # Mots à ignorer
        stop_words = {"le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "à", "au"}
        
        # Extraire les mots
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filtrer les mots courts et les stop words
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        
        return keywords


def create_trigger_analysis_node(llm_manager: LLMManager):
    """
    Crée un node LangGraph pour l'analyse des triggers
    
    Args:
        llm_manager: Gestionnaire LLM à utiliser
        
    Returns:
        Fonction node pour LangGraph
    """
    
    def analyze_triggers_node(state: CharacterState) -> CharacterState:
        """Node qui analyse les triggers d'input"""
        
        # Récupérer les données nécessaires
        user_message = state["user_message"]
        character_data = state["character_data"]
        player_data = state["player_data"]
        
        # Récupérer les triggers d'input
        triggers = character_data.get("triggers", {}).get("input", {})
        
        if not triggers:
            # Pas de triggers définis
            state["input_trigger_probs"] = {}
            state["activated_input_triggers"] = []
            state["refused_input_triggers"] = []
            return state
        
        # Créer et exécuter l'agent
        agent = TriggerAnalysisAgent(llm_manager)
        result = agent.analyze_triggers(
            user_message=user_message,
            triggers=triggers,
            player_data=player_data,
            character_data=character_data
        )
        
        # Mettre à jour le state
        state["input_trigger_probs"] = result.input_trigger_probs
        state["activated_input_triggers"] = result.activated_input_triggers
        state["refused_input_triggers"] = result.refused_input_triggers
        
        # Ajouter à l'historique de traitement
        state["processing_steps"].append("trigger_analysis")
        
        # Debug info
        state["debug_info"]["trigger_analysis"] = {
            "triggers_evaluated": len(triggers),
            "triggers_activated": len(result.activated_input_triggers),
            "triggers_refused": len(result.refused_input_triggers),
            "probabilities": result.input_trigger_probs
        }
        
        return state
    
    return analyze_triggers_node


# Version simplifiée sans agent pour les cas où ReAct est overkill
def create_simple_trigger_analysis_node(llm_manager: LLMManager):
    """
    Version simplifiée du node d'analyse des triggers (sans agent ReAct)
    Plus rapide mais moins flexible
    """
    
    def simple_analyze_triggers_node(state: CharacterState) -> CharacterState:
        """Node simplifié qui analyse les triggers d'input"""
        
        user_message = state["user_message"]
        character_data = state["character_data"]
        player_data = state["player_data"]
        triggers = character_data.get("triggers", {}).get("input", {})
        
        if not triggers:
            state["input_trigger_probs"] = {}
            state["activated_input_triggers"] = []
            state["refused_input_triggers"] = []
            return state
        
        # Construction du prompt pour le LLM
        prompt = f"""Analyse le message utilisateur et détermine les probabilités pour chaque trigger.

Message utilisateur: "{user_message}"

Triggers disponibles:
{json.dumps(triggers, indent=2, ensure_ascii=False)}

Pour chaque trigger, donne une probabilité entre 0.0 et 1.0 que l'intention soit présente dans le message.

Retourne UNIQUEMENT un JSON au format:
{{
    "trigger_name1": 0.0,
    "trigger_name2": 0.5,
    ...
}}"""
        
        # Obtenir les probabilités du LLM
        llm = llm_manager.get_llm()
        response = llm.invoke(prompt)
        
        try:
            # Parser la réponse
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Extraire le JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                probabilities = json.loads(json_match.group())
            else:
                probabilities = {name: 0.0 for name in triggers}
        except:
            probabilities = {name: 0.0 for name in triggers}
        
        # Évaluer les triggers
        activated = []
        refused = []
        
        for trigger_name, trigger_data in triggers.items():
            prob = probabilities.get(trigger_name, 0.0)
            threshold = trigger_data.get("threshold", 0.5)
            
            if prob < threshold:
                refused.append({
                    "trigger": trigger_name,
                    "reason_refused": f"Probabilité ({prob:.2f}) inférieure au seuil ({threshold})"
                })
            else:
                # Vérifier les conditions si présentes
                conditions = trigger_data.get("conditions")
                if conditions:
                    # Évaluation simple des conditions
                    if not _evaluate_simple_condition(conditions, player_data, character_data):
                        refused.append({
                            "trigger": trigger_name,
                            "reason_refused": f"Condition non remplie: {conditions}"
                        })
                    else:
                        activated.append(trigger_name)
                else:
                    activated.append(trigger_name)
        
        # Mettre à jour le state
        state["input_trigger_probs"] = probabilities
        state["activated_input_triggers"] = activated
        state["refused_input_triggers"] = refused
        
        state["processing_steps"].append("simple_trigger_analysis")
        
        return state
    
    return simple_analyze_triggers_node


def _evaluate_simple_condition(condition: str, player_data: Dict, character_data: Dict) -> bool:
    """Évaluation simple de conditions courantes"""
    
    try:
        # Remplacements simples pour les conditions communes
        condition_lower = condition.lower()
        
        # Vérifications d'inventaire
        if "in possession" in condition_lower or "in inventory" in condition_lower:
            item = condition_lower.split()[0]
            inventory = player_data.get("player_stats", {})
            return inventory.get(item, 0) > 0
        
        # Relations
        if "relation" in condition_lower:
            relation = character_data.get("relation", 0)
            # Parser des conditions simples comme "relation > 5"
            if ">" in condition:
                value = int(re.search(r'>\s*(\d+)', condition).group(1))
                return relation > value
            elif "<" in condition:
                value = int(re.search(r'<\s*(\d+)', condition).group(1))
                return relation < value
        
        # États du personnage
        if "is drunk" in condition_lower:
            return character_data.get("personality", {}).get("current_alcohol_level") == "drunk"
        
        return True  # Par défaut, on considère la condition remplie
        
    except:
        return False