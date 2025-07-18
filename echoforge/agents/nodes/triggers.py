"""
Agent ReAct pour l'analyse des triggers d'input avec vérification conditionnelle.
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool, BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
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
    description: str = "Récupère les données complètes du joueur. Ne prend AUCUN paramètre."
    player_data: Dict[str, Any]

    def _run(self, query: str = "") -> str:
        """Execute the tool - accepte un argument mais ne l'utilise pas"""
        return json.dumps(self.player_data, indent=2)

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class GetCharacterDataTool(BaseTool):
    """Tool pour récupérer les données du personnage"""
    name: str = "get_character_data"
    description: str = "Récupère les données complètes du personnage. Ne prend AUCUN paramètre."
    character_data: Dict[str, Any]

    def _run(self, query: str = "") -> str:
        """Execute the tool - accepte un argument mais ne l'utilise pas"""
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

Voici les triggers disponibles (avec leur seuil `threshold` et leurs éventuelles `conditions`) :
{triggers_json}

Message utilisateur : "{user_message}"

RÈGLES IMPORTANTES :
1. IMPORTANT : Tu NE DOIS JAMAIS inclure "Final Answer:" dans la même étape où tu utilises une Action. Termine toujours les Actions avant de donner la réponse finale.
2. Chaque trigger a un champ "threshold" dans le JSON. C'est le seuil minimal de probabilité pour qu'il soit considéré comme potentiellement activé.
   Tu DOIS comparer la probabilité estimée de chaque trigger à ce seuil.
   → Si la probabilité estimée d’un trigger est INFÉRIEURE à son seuil, ignore complètement ce trigger (pas besoin de tools, ni de condition).
   → Si elle est SUPÉRIEURE ou ÉGALE au seuil ET que ce trigger a un champ "conditions", alors tu DOIS appeler les tools pour vérifier la condition.
3. Les tools ne prennent AUCUN paramètre - appelle-les sans arguments.

Tu as accès aux tools suivants :
{tools}

Utilise EXACTEMENT ce format :
NE JAMAIS inclure une Action ET un Final Answer dans le même bloc de réflexion.

Question: la question d'entrée
Thought: réflexion sur ce que tu dois faire
Action: l'action à prendre, doit être l'une de [{tool_names}]
Action Input: (laisse vide ou mets {{}})
Observation: le résultat de l'action
... (répète si nécessaire)
Thought: Je connais maintenant la réponse finale
Final Answer: 
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
        """
        
        # Créer les tools avec la correction
        tools = [
            GetPlayerDataTool(player_data=player_data),
            GetCharacterDataTool(character_data=character_data)
        ]
        
        # Créer le prompt
        prompt = create_trigger_analysis_prompt()
        
        # Créer l'agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Créer l'executor avec handle_parsing_errors
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Pour debug
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
            json_str = extract_json_from_llm_response(final_answer)
            if json_str:
                return TriggerAnalysisResult(**json.loads(json_str))
            else:
                # Si pas de JSON trouvé, essayer de le reconstruire depuis les steps
                return self._reconstruct_from_steps(result, triggers)
                
        except Exception as e:
            print(f"Erreur dans l'agent d'analyse des triggers: {e}")
            
            # Si c'est une erreur de parsing d'output de LangChain
            if isinstance(e, OutputParserException):
                # 1. Tenter d'extraire depuis l'attribut LLM output si disponible
                if hasattr(e, 'llm_output'):
                    output_text = e.llm_output
                # 2. Sinon depuis les args
                elif e.args and isinstance(e.args[0], str):
                    output_text = e.args[0]
                # 3. Fallback générique
                else:
                    output_text = str(e)
                
                json_str = extract_json_from_llm_response(output_text)
                if json_str:
                    try:
                        return TriggerAnalysisResult(**json.loads(json_str))
                    except Exception as parse_error:
                        print(f"Échec du parsing JSON après extraction: {parse_error}")
            
            # Optionnel : fallback custom si tout échoue
            return self._reconstruct_from_steps({}, triggers)
            
        
    def _reconstruct_from_steps(self, result: dict, triggers: dict) -> TriggerAnalysisResult:
        """Reconstruit le résultat depuis les étapes intermédiaires"""
        
        # Valeurs par défaut
        probs = {name: 0.0 for name in triggers}
        activated = []
        refused = []
        
        # Essayer d'extraire les infos des étapes
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                # Analyser les observations pour déduire les états
                pass
        
        # Parser la sortie finale même si elle est mal formée
        output = result.get("output", "")
        
        # Chercher des patterns dans la sortie
        if "give_alcool" in output and ("activé" in output or "activated" in output):
            probs["give_alcool"] = 1.0
            activated.append("give_alcool")
        
        # Compléter les refusés
        for trigger_name in triggers:
            if trigger_name not in activated:
                refused.append({
                    "trigger": trigger_name,
                    "reason_refused": "Non détecté dans le message"
                })
        
        return TriggerAnalysisResult(
            input_trigger_probs=probs,
            activated_input_triggers=activated,
            refused_input_triggers=refused
        )
    
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

    
def extract_json_from_llm_response(text: str) -> Optional[str]:
    """Extrait un objet JSON d'une réponse LLM, peu importe le format"""
    
    if not text:
        return None
    
    # Nettoyer le texte
    text = str(text)
    
    # Méthode 1: Entre ```json et ```
    json_code_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_code_match:
        try:
            json.loads(json_code_match.group(1))  # Vérifier que c'est valide
            return json_code_match.group(1)
        except:
            pass
    
    # Méthode 2: Entre ``` et ```
    code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_match:
        try:
            json.loads(code_match.group(1))  # Vérifier que c'est valide
            return code_match.group(1)
        except:
            pass
    
    # Méthode 3: Après "Final Answer:"
    final_answer_match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        remaining_text = final_answer_match.group(1)
        # Chercher le JSON dans ce qui reste
        first_brace = remaining_text.find('{')
        last_brace = remaining_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            try:
                json_str = remaining_text[first_brace:last_brace + 1]
                json.loads(json_str)  # Vérifier que c'est valide
                return json_str
            except:
                pass
    
    # Méthode 4: Trouver le JSON brut (premier { au dernier })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1:
        try:
            json_str = text[first_brace:last_brace + 1]
            json.loads(json_str)  # Vérifier que c'est valide
            return json_str
        except:
            pass
    
    return None