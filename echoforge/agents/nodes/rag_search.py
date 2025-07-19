from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import create_react_agent, AgentExecutor, AgentType
from langchain.tools import Tool, BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field
import json
from echoforge.agents.state.character_state import CharacterState
from echoforge.core.llm_providers import LLMManager
from echoforge.core import EchoForgeRAG
from echoforge.utils.config import get_config
from langsmith import traceable


class SearchWorldKnowledgeTool(BaseTool):
    """Tool pour rechercher dans les connaissances du monde"""
    name: str = "search_world"
    description: str = "Recherche dans les connaissances générales du monde et de l'île. Utilise pour : histoire, lieux, événements globaux."
    rag_system: EchoForgeRAG
    evaluate_tool: Optional['EvaluateRelevanceTool'] = None
    
    def _run(self, query: str) -> str:
        """Execute la recherche dans le monde"""
        try:
            results = []
            
            # Recherche uniquement dans les connaissances du monde
            world_context = self.rag_system.retrieve_world_context(query, top_k=5)
            for i, content in enumerate(world_context):
                results.append({
                    "content": content,
                    "source": "world_knowledge",
                    "relevance": "high" if i == 0 else "medium"
                })
            
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            
            # Stocker les résultats pour l'outil d'évaluation
            if self.evaluate_tool:
                # Ajouter aux résultats existants ou créer
                if hasattr(self.evaluate_tool, 'last_results') and self.evaluate_tool.last_results:
                    try:
                        existing = json.loads(self.evaluate_tool.last_results)
                        existing.extend(results)
                        self.evaluate_tool.last_results = json.dumps(existing, indent=2, ensure_ascii=False)
                    except:
                        self.evaluate_tool.last_results = results_json
                else:
                    self.evaluate_tool.last_results = results_json
            
            return results_json
            
        except Exception as e:
            return f"Erreur lors de la recherche monde: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class SearchCharacterKnowledgeTool(BaseTool):
    """Tool pour rechercher dans les connaissances du personnage"""
    name: str = "search_character"
    description: str = "Recherche dans les connaissances personnelles du personnage. Utilise pour : souvenirs, relations, secrets personnels, enfance."
    rag_system: EchoForgeRAG
    character_name: str
    evaluate_tool: Optional['EvaluateRelevanceTool'] = None
    
    def _run(self, query: str) -> str:
        """Execute la recherche sur le personnage"""
        try:
            results = []
            
            # Recherche uniquement dans les connaissances du personnage
            character_context = self.rag_system.retrieve_character_context(
                query, self.character_name.lower(), top_k=5
            )
            for i, content in enumerate(character_context):
                results.append({
                    "content": content,
                    "source": f"{self.character_name}_knowledge",
                    "relevance": "high" if i == 0 else "medium"
                })
            
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            
            # Stocker les résultats pour l'outil d'évaluation
            if self.evaluate_tool:
                # Ajouter aux résultats existants ou créer
                if hasattr(self.evaluate_tool, 'last_results') and self.evaluate_tool.last_results:
                    try:
                        existing = json.loads(self.evaluate_tool.last_results)
                        existing.extend(results)
                        self.evaluate_tool.last_results = json.dumps(existing, indent=2, ensure_ascii=False)
                    except:
                        self.evaluate_tool.last_results = results_json
                else:
                    self.evaluate_tool.last_results = results_json
            
            return results_json
            
        except Exception as e:
            return f"Erreur lors de la recherche personnage: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class EvaluateRelevanceTool(BaseTool):
    """Tool pour évaluer la pertinence des résultats"""
    name: str = "evaluate_relevance"
    description: str = "Évalue si les derniers résultats de recherche sont suffisants pour répondre. Ne prend aucun paramètre."
    last_results: Optional[str] = None
    
    def _run(self, query: str = "") -> str:
        """Évalue la pertinence des derniers résultats"""
        if not self.last_results:
            return "error - aucun résultat à évaluer"
            
        try:
            results = json.loads(self.last_results)
            if not results:
                return "insufficient - aucun résultat trouvé"
            
            # Analyse basique pour aider l'agent
            has_high_relevance = any(r.get("relevance") == "high" for r in results)
            has_character_info = any("_knowledge" in r.get("source", "") for r in results)
            has_world_info = any(r.get("source") == "world_knowledge" for r in results)
            has_content = all(r.get("content", "").strip() for r in results)
            
            if has_high_relevance and has_content:
                info_type = []
                if has_character_info:
                    info_type.append("personnage")
                if has_world_info:
                    info_type.append("monde")
                return f"sufficient - informations pertinentes trouvées ({' et '.join(info_type)})"
            elif has_content:
                return "partial - informations trouvées mais pertinence moyenne"
            else:
                return "insufficient - informations insuffisantes ou non pertinentes"
                
        except Exception as e:
            return f"error - impossible d'évaluer: {str(e)}"
    
    def reset(self):
        """Réinitialise les résultats stockés"""
        self.last_results = None
    
    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class AnalyzeQuestionTool(BaseTool):
    """Tool pour analyser la question et extraire les concepts clés"""
    name: str = "analyze_question"
    description: str = "Analyse la question pour identifier les concepts clés et le type d'information recherchée."
    
    def _run(self, question: str) -> str:
        """Analyse la question"""
        # Analyse simple des mots-clés et patterns
        keywords = []
        question_lower = question.lower()
        
        # Patterns de questions
        patterns = {
            "historical": ["histoire", "passé", "avant", "autrefois", "origine"],
            "relationship": ["relation", "ami", "ennemi", "famille", "connais"],
            "location": ["où", "lieu", "endroit", "trouve", "situé"],
            "personality": ["caractère", "personnalité", "comportement", "pourquoi", "toi", "enfance"],
            "action": ["faire", "peut", "capable", "donne", "aide"],
            "knowledge": ["sais", "connais", "explique", "raconte", "parle"]
        }
        
        detected_types = []
        for ptype, words in patterns.items():
            if any(word in question_lower for word in words):
                detected_types.append(ptype)
        
        # Extraction de mots-clés importants
        important_words = [w for w in question.split() if len(w) > 3 and w.lower() not in ["pour", "avec", "dans", "mais"]]
        
        # Détection spéciale pour questions personnelles
        needs_personal = any(word in question_lower for word in ["toi", "ton", "ta", "tes", "enfance", "passé"])
        
        return json.dumps({
            "question_types": detected_types,
            "keywords": important_words[:5],
            "needs_personal_info": needs_personal,
            "needs_world_info": "location" in detected_types or "historical" in detected_types
        }, ensure_ascii=False)
    
    async def _arun(self, question: str) -> str:
        return self._run(question)


def create_rag_agent_prompt() -> PromptTemplate:
    """Crée le prompt pour l'agent RAG ReAct"""
    
    template = """Tu es un assistant de recherche intelligent pour le personnage {character_name}.
Ta mission est de déterminer si une recherche dans la base de connaissances est nécessaire et, si oui, de trouver les informations pertinentes.

Personnage actuel: {character_name}
Question de l'utilisateur: "{user_message}"
Intention détectée: {intent}

Tu as accès aux outils suivants:
{tools}

RÈGLES IMPORTANTES:
1. Pour les salutations et questions simples, tu peux directement donner la Final Answer sans utiliser d'outils
2. N'utilise les outils QUE si la question nécessite des connaissances spécifiques
3. Après avoir analysé la question, si c'est une salutation ou une question simple, passe directement à la Final Answer

PROCESSUS DE DÉCISION:

1. D'abord, analyse mentalement si la question nécessite une recherche:
   - Salutations ("bonjour", "salut", etc.) → Final Answer directe avec needs_rag: false
   - Questions simples (oui/non, émotions basiques) → Final Answer directe avec needs_rag: false
   - Questions sur l'histoire, relations, lieux, événements → Utilise les outils
   - Questions sur le personnage, son passé, ses connaissances → Utilise les outils

2. Si recherche nécessaire:
   - Utilise analyze_question pour comprendre la question
   - Choisis le(s) bon(s) outil(s):
     * search_world pour : histoire de l'île, lieux, événements globaux
     * search_character pour : souvenirs personnels, enfance, relations, secrets
     * Tu peux utiliser les DEUX si la question touche aux deux aspects
   - Utilise evaluate_relevance (sans paramètres) après TOUTES les recherches
   - Si insuffisant, reformule et recherche à nouveau (max 3 tentatives)

3. Format de réponse:
   - Si aucune recherche effectuée → relevant_knowledge: []
   - Si recherche effectuée → inclure TOUS les contenus pertinents trouvés

Utilise EXACTEMENT ce format:
Question: la question d'entrée
Thought: réflexion sur ce que tu dois faire
Action: l'action à prendre, doit être l'une de [{tool_names}]
Action Input: l'input de l'action
Observation: le résultat de l'action
... (répète si nécessaire)
Thought: Je connais maintenant la réponse finale
Final Answer: {{
    "needs_rag": true/false,
    "search_performed": true/false,
    "relevant_knowledge": [],
    "search_queries": [],
    "evaluation": "none",
    "reasoning": "explication courte"
}}

EXEMPLE pour une salutation:
Question: Bonjour !
Thought: C'est une salutation simple qui ne nécessite pas de recherche dans la base de connaissances.
Final Answer: {{
    "needs_rag": false,
    "search_performed": false,
    "relevant_knowledge": [],
    "search_queries": [],
    "evaluation": "none",
    "reasoning": "Salutation simple ne nécessitant pas de recherche"
}}

EXEMPLE pour une question personnelle:
Question: Parle-moi de ton enfance
Thought: L'utilisateur demande des informations personnelles sur mon enfance. Je dois chercher dans mes connaissances personnelles.
Action: analyze_question
Action Input: "Parle-moi de ton enfance"
Observation: {{"question_types": ["personality", "knowledge"], "keywords": ["Parle-moi", "enfance"], "needs_personal_info": true, "needs_world_info": false}}

Thought: L'analyse confirme que c'est une question personnelle. Je dois rechercher dans mes souvenirs personnels.
Action: search_character
Action Input: "enfance passé souvenirs famille jeunesse"
Observation: [résultats sur l'enfance du personnage]

Thought: J'ai trouvé des informations sur mon enfance. Je dois évaluer si c'est suffisant.
Action: evaluate_relevance
Action Input: 
Observation: sufficient - informations pertinentes trouvées (personnage)

Thought: Je connais maintenant la réponse finale
Final Answer: {{
    "needs_rag": true,
    "search_performed": true,
    "relevant_knowledge": ["contenu sur l'enfance trouvé"],
    "search_queries": ["enfance passé souvenirs famille jeunesse"],
    "evaluation": "sufficient",
    "reasoning": "Question personnelle sur l'enfance nécessitant des recherches dans les souvenirs du personnage"
}}

EXEMPLE pour une question mixte:
Question: Comment ton enfance sur l'île t'a-t-elle influencé?
Thought: Cette question touche à la fois l'enfance personnelle ET l'histoire de l'île. Je dois rechercher dans les deux.
Action: search_character
Action Input: "enfance influence personnalité formation"
Observation: [résultats personnels]

Thought: J'ai des infos personnelles, mais je dois aussi chercher sur l'île.
Action: search_world  
Action Input: "île histoire événements période enfance"
Observation: [résultats sur l'île]

Thought: J'ai maintenant des infos des deux sources. Vérifions si c'est suffisant.
Action: evaluate_relevance
Action Input:
Observation: sufficient - informations pertinentes trouvées (personnage et monde)

Thought: Je connais maintenant la réponse finale
Final Answer: {{
    "needs_rag": true,
    "search_performed": true,
    "relevant_knowledge": ["infos enfance personnage", "contexte historique île"],
    "search_queries": ["enfance influence personnalité formation", "île histoire événements période enfance"],
    "evaluation": "sufficient",
    "reasoning": "Question mixte nécessitant des recherches personnelles et contextuelles"
}}

Commence!

Question: {user_message}
{agent_scratchpad}"""
    
    return PromptTemplate(
        template=template,
        input_variables=["character_name", "user_message", "intent", "tools", "tool_names", "agent_scratchpad"]
    )


class ReactRAGAgent:
    """Agent ReAct pour gérer intelligemment les recherches RAG"""
    
    def __init__(self, llm_manager: LLMManager, rag_system: EchoForgeRAG):
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_llm()
        self.rag_system = rag_system
        self.config = get_config()
    
    @traceable
    def process_rag_need(
        self,
        user_message: str,
        character_name: str,
        intent: str,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Traite le besoin de RAG et effectue les recherches si nécessaire.
        
        Returns:
            Dict avec:
            - needs_rag: bool
            - search_performed: bool
            - relevant_knowledge: List[str]
            - search_queries: List[str]
            - evaluation: str
            - reasoning: str
        """
        
        # Créer les tools
        evaluate_tool = EvaluateRelevanceTool()
        
        tools = [
            AnalyzeQuestionTool(),
            SearchWorldKnowledgeTool(
                rag_system=self.rag_system,
                evaluate_tool=evaluate_tool
            ),
            SearchCharacterKnowledgeTool(
                rag_system=self.rag_system, 
                character_name=character_name,
                evaluate_tool=evaluate_tool
            ),
            evaluate_tool
        ]
        
        # Créer le prompt
        prompt = create_rag_agent_prompt()
        
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
            max_iterations=4,  # Réduit pour éviter les boucles
            handle_parsing_errors=True,
            max_execution_time=30  # Timeout de 30 secondes
        )
        
        # Exécuter l'agent
        try:
            result = agent_executor.invoke({
                "character_name": character_name,
                "user_message": user_message,
                "intent": intent or "general"
            })
            
            # Parser le résultat
            final_answer = result.get("output", "{}")
            
            # Extraire le JSON de la réponse
            import re
            json_match = re.search(r'\{.*\}', final_answer, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                # S'assurer que tous les champs sont présents
                return {
                    "needs_rag": parsed_result.get("needs_rag", False),
                    "search_performed": parsed_result.get("search_performed", False),
                    "relevant_knowledge": parsed_result.get("relevant_knowledge", []),
                    "search_queries": parsed_result.get("search_queries", []),
                    "evaluation": parsed_result.get("evaluation", "none"),
                    "reasoning": parsed_result.get("reasoning", "Aucune recherche nécessaire")
                }
            else:
                # Fallback si pas de JSON trouvé
                return {
                    "needs_rag": False,
                    "search_performed": False,
                    "relevant_knowledge": [],
                    "search_queries": [],
                    "evaluation": "none",
                    "reasoning": "Pas de recherche effectuée (parsing error)"
                }
                
        except Exception as e:
            print(f"Erreur dans l'agent RAG ReAct: {e}")
            return {
                "needs_rag": False,
                "search_performed": False,
                "relevant_knowledge": [],
                "search_queries": [],
                "evaluation": "error",
                "reasoning": f"Erreur: {str(e)}"
            }


def create_react_rag_node(llm_manager: LLMManager):
    """
    Crée un node LangGraph qui utilise l'agent ReAct pour le RAG.
    Remplace assess_rag_need + rag_search + validate_rag_results.
    
    Args:
        llm_manager: Gestionnaire LLM à utiliser
        
    Returns:
        Fonction node pour LangGraph
    """
    
    @traceable
    def react_rag_node(state: CharacterState) -> CharacterState:
        """Node qui gère tout le processus RAG avec un agent ReAct"""
        
        state["processing_steps"].append("react_rag_agent")
        
        # Initialiser le système RAG
        config = get_config()
        rag_system = EchoForgeRAG(
            data_path=str(config.data_path),
            vector_store_path=str(config.vector_store_path),
            embedding_model=config.embedding_model,
            llm_model=config.llm_model
        )
        
        # Créer l'agent
        agent = ReactRAGAgent(llm_manager, rag_system)
        
        # Traiter avec l'agent
        result = agent.process_rag_need(
            user_message=state["parsed_message"],
            character_name=state["character_name"],
            intent=state["message_intent"],
            conversation_history=state.get("conversation_history", [])
        )
        
        # Mettre à jour le state avec les résultats
        state["needs_rag_search"] = result["needs_rag"]
        state["rag_query"] = result.get("search_queries", [])
        
        # Convertir les résultats en format attendu
        if result["search_performed"] and result["relevant_knowledge"]:
            rag_results = []
            for i, content in enumerate(result["relevant_knowledge"]):
                rag_results.append({
                    "content": content,
                    "metadata": {"type": "react_search"},
                    "relevance": 1.0 - (i * 0.1),  # Score décroissant
                    "source": "react_agent"
                })
            state["rag_results"] = rag_results
            state["relevant_knowledge"] = result["relevant_knowledge"]
        else:
            state["rag_results"] = []
            state["relevant_knowledge"] = []
        
        # Debug info
        state["debug_info"]["react_rag_agent"] = {
            "needs_rag": result["needs_rag"],
            "search_performed": result["search_performed"],
            "queries_count": len(result.get("search_queries", [])),
            "results_count": len(result.get("relevant_knowledge", [])),
            "evaluation": result.get("evaluation", "none"),
            "reasoning": result.get("reasoning", "")
        }
        
        return state
    
    return react_rag_node


# """Nœud de recherche RAG."""

# from typing import List, Dict, Any
# from ..state.character_state import CharacterState
# from langsmith import traceable
# from echoforge.core import EchoForgeRAG
# from echoforge.utils.config import get_config

# config = get_config()
# @traceable
# def perform_rag_search(llm_manager):
#     """
#     Effectue une recherche RAG basée sur la requête déterminée.
    
#     Args:
#         state: État actuel du personnage
        
#     Returns:
#         État mis à jour avec les résultats RAG
#     """
#     def fn(state: CharacterState) -> CharacterState:
#         state["processing_steps"].append("rag_search")
#         query = state["rag_query"][-1]

#         if state.get("needs_rag_retry"):
#             new_query = _reformulate_query_with_llm(state, previous_query=query, llm_manager=llm_manager)
#             if new_query:
#                 state["rag_query"].append(new_query)
#                 query = new_query
        
#         character_name = state["character_name"]
#         try:
#             rag_system = EchoForgeRAG(
#                     data_path=str(config.data_path),
#                     vector_store_path=str(config.vector_store_path),
#                     embedding_model=config.embedding_model,
#                     llm_model=config.llm_model
#                 )
#             results = []

#             # Recherche dans les connaissances du monde
#             world_context = rag_system.retrieve_world_context(query, top_k=config.top_k_world)
#             for i, content in enumerate(world_context):
#                 results.append({
#                     "content": content,
#                     "metadata": {"type": "world", "importance": "medium"},
#                     "relevance": max(0.8 - i * 0.1, 0.3),  # Score décroissant
#                     "source": "world_knowledge"
#                 })
            
#             # Recherche dans les connaissances du personnage
#             character_context = rag_system.retrieve_character_context(
#                 query, character_name.lower(), top_k=config.top_k_character
#             )
#             for i, content in enumerate(character_context):
#                 results.append({
#                     "content": content,
#                     "metadata": {"type": "character", "importance": "high"},
#                     "relevance": max(0.9 - i * 0.1, 0.4),  # Score plus élevé pour le personnage
#                     "source": f"{character_name}_knowledge"
#                 })
            
#             # Trie par pertinence décroissante
#             results.sort(key=lambda x: x["relevance"], reverse=True)
            
#         except ImportError as e:
#             print("⚠️ EchoForgeRAG non disponible, utilisation de la simulation")
#             return state
        
#         except Exception as e:
#             print(f"⚠️ Erreur lors de la recherche RAG: {e}")
#             return state
        
#         # Limite à 5 résultats maximum
#         rag_results = results[:5]
        
#         # Mise à jour de l'état
#         state["rag_results"].extend(rag_results)
        
#         # Debug info
#         state["debug_info"]["rag_search"] = {
#             "query": query,
#             "results_count": len(rag_results),
#             "top_relevance_score": rag_results[0]["relevance"] if rag_results else 0
#         }
        
#         return state
#     return fn

# def _reformulate_query_with_llm(state: CharacterState, previous_query: str, llm_manager) -> str:
#     """
#     Reformule une requête RAG plus efficace à partir du message utilisateur,
#     de l’intention, et du contexte de recherche précédent.
#     """
#     user_msg = state.get("parsed_message") or state.get("user_message", "")
#     intent = state.get("message_intent", "")
#     character_name = state.get("character_name", "le personnage")
#     rag_results = state.get("rag_results", [])
    
#     previous_knowledge = "\n".join(
#         f"- {r['content']}" for r in rag_results[:3]
#     ) if rag_results else "Aucun résultat pertinent précédemment trouvé."

#     prompt = f"""
# Tu es un assistant expert en recherche de connaissances narratives pour un jeu de rôle.

# Le personnage s'appelle {character_name}.

# Le joueur a dit : "{user_msg}"
# Intention détectée : {intent}
# Ancienne requête utilisée : "{previous_query}"

# Voici les résultats précédemment trouvés :
# {previous_knowledge}

# Tu dois générer une nouvelle requête optimisée, plus précise, qui aiderait le personnage à trouver des informations pertinentes.

# Réponds uniquement par la requête reformulée, sans autre texte.
# Si aucune reformulation pertinente n'est possible, réponds exactement : NONE
# """

#     try:
#         new_query = llm_manager.invoke(prompt).strip()
#         if new_query.upper() == "NONE":
#             return None
#         return new_query
#     except Exception as e:
#         state["debug_info"]["query_reformulation_error"] = str(e)
#         return None