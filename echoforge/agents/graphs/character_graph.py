"""Graphe principal pour les personnages EchoForge utilisant LangGraph."""

from langgraph.graph import StateGraph, END
from ..state.character_state import CharacterState
from ..nodes.perception import perceive_input
from ..nodes.rag_assessment import assess_rag_need
from ..nodes.rag_search import perform_rag_search
from ..nodes.response_generation import generate_simple_response, generate_response
from ..nodes.memory_update import update_character_memory, finalize_interaction
from langsmith import traceable
from ..conditions.complexity_router import (
    route_by_complexity, 
    route_by_rag_need, 
    check_if_needs_memory_update
)
from echoforge.utils.config import get_config

config = get_config()

def create_character_graph() -> StateGraph:
    """
    Crée le graphe principal d'un personnage EchoForge.
    
    Ce graphe gère le flux complet depuis la perception d'un message
    jusqu'à la génération de la réponse finale avec mise à jour de la mémoire.
    
    Returns:
        StateGraph: Graphe compilé prêt à être exécuté
    """
    
    # === CRÉATION DU GRAPHE ===
    graph = StateGraph(CharacterState)
    
    # === AJOUT DES NŒUDS ===
    
    # Nœud d'entrée : perception et analyse du message
    graph.add_node("perceive", perceive_input)
    
    # Nœuds de réponse selon la complexité
    graph.add_node("simple_response", generate_simple_response)
    graph.add_node("assess_rag_need", assess_rag_need)
    graph.add_node("rag_search", perform_rag_search)
    graph.add_node("generate_response", generate_response)
    
    # Nœuds de finalisation
    graph.add_node("memory_update", update_character_memory)
    graph.add_node("finalize", finalize_interaction)
    
    # === DÉFINITION DU POINT D'ENTRÉE ===
    graph.set_entry_point("perceive")
    
    # === DÉFINITION DES FLUX ===
    
    # Depuis la perception, routage selon la complexité
    graph.add_conditional_edges(
        "perceive",
        route_by_complexity,
        {
            "simple_response": "simple_response",
            "assess_rag_need": "assess_rag_need"
        }
    )
    
    # Depuis l'évaluation RAG, routage selon le besoin
    graph.add_conditional_edges(
        "assess_rag_need", 
        route_by_rag_need,
        {
            "rag_search": "rag_search",
            "generate_response": "generate_response"
        }
    )
    
    # Depuis la recherche RAG, vers la génération de réponse
    graph.add_edge("rag_search", "generate_response")
    
    # Depuis les réponses, routage vers mémoire ou finalisation
    graph.add_conditional_edges(
        "simple_response",
        check_if_needs_memory_update,
        {
            "memory_update": "memory_update",
            "finalize": "finalize"
        }
    )
    
    graph.add_conditional_edges(
        "generate_response",
        check_if_needs_memory_update,
        {
            "memory_update": "memory_update", 
            "finalize": "finalize"
        }
    )
    
    # Depuis la mise à jour mémoire, vers la finalisation
    graph.add_edge("memory_update", "finalize")
    
    # Depuis la finalisation, fin du graphe
    graph.add_edge("finalize", END)
    
    return graph


def create_simple_chat_graph() -> StateGraph:
    """
    Crée un graphe simplifié pour les conversations rapides.
    
    Ce graphe optimisé évite les étapes complexes pour les interactions simples
    comme les salutations ou réponses courtes.
    
    Returns:
        StateGraph: Graphe simplifié compilé
    """
    
    graph = StateGraph(CharacterState)
    
    # Nœuds simplifiés
    graph.add_node("quick_perceive", perceive_input)
    graph.add_node("quick_response", generate_simple_response)  
    graph.add_node("quick_finalize", finalize_interaction)
    
    # Flux linéaire simplifié
    graph.set_entry_point("quick_perceive")
    graph.add_edge("quick_perceive", "quick_response")
    graph.add_edge("quick_response", "quick_finalize")
    graph.add_edge("quick_finalize", END)
    
    return graph


class CharacterGraphManager:
    """
    Gestionnaire pour les graphes de personnages.
    
    Permet de sélectionner automatiquement le bon graphe selon le contexte
    et de gérer l'exécution avec configuration appropriée.
    """
    
    def __init__(self):
        self.main_graph = create_character_graph()
        self.simple_graph = create_simple_chat_graph()
        self.compiled_main = self.main_graph.compile()
        self.compiled_simple = self.simple_graph.compile()

    @traceable
    async def process_message(
        self, 
        user_message: str, 
        character_data: dict,
        use_simple_mode: bool = False,
        thread_id: str = "default"
    ) -> dict:
        """
        Traite un message utilisateur avec le graphe approprié.
        
        Args:
            user_message: Message de l'utilisateur
            character_data: Données du personnage
            use_simple_mode: Force l'utilisation du graphe simple
            thread_id: ID pour la persistance de conversation
            
        Returns:
            État final avec la réponse générée
        """
        
        # Construction de l'état initial
        initial_state = self._build_initial_state(user_message, character_data)
        
        # Configuration pour LangGraph
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Sélection du graphe
        if use_simple_mode or self._should_use_simple_graph(user_message):
            result = await self.compiled_simple.ainvoke(initial_state, config=config)
        else:
            result = await self.compiled_main.ainvoke(initial_state, config=config)
        
        return result
    
    @traceable
    def _build_initial_state(self, user_message: str, character_data: dict) -> CharacterState:
        """Construit l'état initial pour le graphe."""
        
        return CharacterState(
            # Input
            user_message=user_message,
            response="",
            
            # Analyse (sera remplie par le graphe)
            parsed_message=None,
            message_intent=None
            
            # Personnage
            character_name=character_data.get("name", "unknown"),
            personality_traits=character_data.get("personality", {}),
            current_emotion=character_data.get("current_emotion", "neutral"),
            character_knowledge=character_data.get("knowledge", []),
            
            # Conversation
            conversation_history=character_data.get("conversation_history", []),
            context_summary=None,
            
            # RAG
            needs_rag_search=False,
            rag_query=None,
            rag_results=[],
            relevant_knowledge=[],
            
            # Actions
            input_trigger_probs=character_data.get("triggers")
            
            # Métadonnées
            processing_start_time=0.0,
            processing_steps=[],
            debug_info={}
        )
    # def _build_initial_state(self, user_message: str, character_data: dict) -> CharacterState:
    #     """Construit l'état initial pour le graphe."""
        
    #     return CharacterState(
    #         # Input
    #         user_message=user_message,
    #         response="",
            
    #         # Analyse (sera remplie par le graphe)
    #         parsed_message=None,
    #         message_intent=None,
    #         complexity_level="medium",
            
    #         # Personnage
    #         character_name=character_data.get("name", "unknown"),
    #         personality_traits=character_data.get("personality", {}),
    #         current_emotion=character_data.get("current_emotion", "neutral"),
    #         character_knowledge=character_data.get("knowledge", []),
            
    #         # Conversation
    #         conversation_history=character_data.get("conversation_history", []),
    #         context_summary=None,
            
    #         # RAG
    #         needs_rag_search=False,
    #         rag_query=None,
    #         rag_results=[],
    #         relevant_knowledge=[],
            
    #         # Actions
    #         planned_actions=[],
    #         triggered_events=[],
    #         game_state_changes={},
            
    #         # Métadonnées
    #         processing_start_time=0.0,
    #         processing_steps=[],
    #         debug_info={}
    #     )
    
    def _should_use_simple_graph(self, user_message: str) -> bool:
        """Détermine si le graphe simple doit être utilisé."""
        
        message_lower = user_message.lower().strip()

        # Patterns simples
        simple_patterns = [
            "bonjour", "salut", "hey", "hello",
            "au revoir", "bye", "à bientôt",
            "merci", "de rien", "ok", "d'accord",
            "oui", "non", "peut-être"
        ]
        
        if len(message_lower) < 10 and any(pattern in message_lower for pattern in simple_patterns):
            return True
        
        return False
    
    def get_graph_visualization(self, graph_type: str = "main") -> str:
        """
        Retourne une représentation textuelle du graphe.
        
        Args:
            graph_type: "main" ou "simple"
            
        Returns:
            Représentation textuelle du graphe
        """
        
        if graph_type == "simple":
            return """
Graphe Simple:
perceive → simple_response → finalize → END
            """
        else:
            return """
Graphe Principal:
perceive → [complexity_router]
├─ simple_response → [memory_router] → finalize/memory_update → END
└─ assess_rag_need → [rag_router]
   ├─ rag_search → generate_response → [memory_router] → finalize/memory_update → END  
   └─ generate_response → [memory_router] → finalize/memory_update → END
            """