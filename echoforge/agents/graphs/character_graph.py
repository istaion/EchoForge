from langgraph.graph import StateGraph, END
from echoforge.agents.state.character_state import CharacterState
from echoforge.agents.nodes.perception import perceive_input, interpret_player_input_node, decide_intent_node, interpret_character_output, interpret_triggers_input_node
from echoforge.agents.nodes.rag_assessment import assess_rag_need, validate_rag_results
from echoforge.agents.nodes.rag_search import perform_rag_search
from echoforge.agents.nodes.response_generation import generate_simple_response, generate_response
from echoforge.agents.nodes.triggers import create_trigger_analysis_node
from echoforge.agents.nodes.memory_update import update_character_memory, finalize_interaction, load_memory_context, check_memory_integration
from echoforge.agents.checkpointers.postgres_checkpointer import create_safe_checkpointer, NoOpCheckpointSaver
from langsmith import traceable
from echoforge.agents.conditions.complexity_router import (
    route_by_complexity, 
    route_by_rag_need, 
    check_if_needs_new_rag
)
from echoforge.core.llm_providers import LLMManager
from echoforge.utils.config import get_config
from sqlmodel import Session, select, and_
from echoforge.db.database import get_session
from typing import Dict, Any, List, Optional
from echoforge.db.models.memory import ConversationSummary
import os
import time

config = get_config()


def create_character_graph_with_memory(character_name: str, enable_checkpointer: bool = True) -> StateGraph:
    """
    Crée le graphe principal d'un personnage avec système de mémoire intégré.
    🆕 Version mise à jour avec support session_id complet.
    
    Args:
        character_name: Nom du personnage pour la persistance
        enable_checkpointer: Active ou désactive le checkpointer
        
    Returns:
        StateGraph: Graphe compilé avec ou sans checkpointer
    """
    
    # === CRÉATION DU GRAPHE ===
    graph = StateGraph(CharacterState)
    
    # === AJOUT DES NŒUDS ===
    
    # 🆕 Nœud d'entrée : chargement du contexte de mémoire avec session_id
    graph.add_node("load_memory", load_memory_context)
    
    # Nœud de perception et analyse du message
    graph.add_node("perceive", perceive_input)
    
    # Analyse des triggers d'input
    graph.add_node("interpret_input", create_trigger_analysis_node(llm_manager=LLMManager()))
    # graph.add_node("interpret_input", interpret_player_input_node(llm_manager=LLMManager()))
    # graph.add_node("decide_intent", decide_intent_node())
    
    # Nœud de vérification de l'intégration mémoire
    graph.add_node("check_memory_integration", check_memory_integration)
    
    # Nœuds de réponse selon la complexité
    graph.add_node("simple_response", generate_simple_response)
    graph.add_node("assess_rag_need", assess_rag_need)
    graph.add_node("rag_search", perform_rag_search(llm_manager=LLMManager()))
    graph.add_node("validate_rag_results", validate_rag_results)
    graph.add_node("generate_response", generate_response)
    
    # Nœuds de finalisation avec système de mémoire
    graph.add_node("interpret_output", interpret_character_output(llm_manager=LLMManager()))
    graph.add_node("memory_update", update_character_memory)
    graph.add_node("finalize", finalize_interaction)
    
    # === DÉFINITION DU POINT D'ENTRÉE ===
    graph.set_entry_point("load_memory")
    
    # === DÉFINITION DES FLUX ===
    
    # Flux avec chargement de mémoire
    graph.add_edge("load_memory", "interpret_input")
    # graph.add_edge("interpret_input", "decide_intent")
    graph.add_edge("interpret_input", "perceive")
    graph.add_edge("perceive", "check_memory_integration")
    
    # Depuis la vérification mémoire, routage selon la complexité
    graph.add_conditional_edges(
        "check_memory_integration",
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
    graph.add_edge("rag_search", "validate_rag_results")
    graph.add_conditional_edges(
        "validate_rag_results",
        check_if_needs_new_rag,
        {
            "rag_retry": "rag_search",
            "generate_response": "generate_response"
        }
    )
    
    # Depuis les réponses, routage vers mémoire ou finalisation
    graph.add_edge("simple_response", "interpret_output")
    graph.add_edge("generate_response", "interpret_output")
    graph.add_edge("interpret_output", "memory_update")
    
    # Depuis la mise à jour mémoire, vers la finalisation
    graph.add_edge("memory_update", "finalize")
    
    # Depuis la finalisation, fin du graphe
    graph.add_edge("finalize", END)
    
    # === COMPILATION AVEC CHECKPOINTER SÉCURISÉ ===
    try:
        # Crée un checkpointer sûr avec fallback automatique
        checkpointer = create_safe_checkpointer(
            character_name=character_name,
            enable_checkpointer=enable_checkpointer
        )
        
        # Compile le graphe avec le checkpointer
        compiled_graph = graph.compile(checkpointer=checkpointer)
        
        # Teste la compilation
        print(f"✅ Graphe compilé avec succès pour {character_name} (avec récupération mémoire + session)")
        
        return compiled_graph
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la compilation avec checkpointer: {e}")
        print("📝 Compilation sans checkpointer comme fallback")
        
        # Fallback : compilation sans checkpointer
        return graph.compile()


class CharacterGraphManager:
    """
    Gestionnaire pour les graphes de personnages avec mémoire persistante.
    🆕 Version améliorée avec support session_id complet.
    """
    
    def __init__(self, enable_checkpointer: bool = True):
        self._character_graphs = {}
        self.enable_checkpointer = enable_checkpointer
        self._fallback_mode = False
        
        # Teste la disponibilité du système de base de données
        self._test_database_connection()
    
    def _test_database_connection(self):
        """Teste la connexion à la base de données."""
        try:
            with get_session() as session:
                session.exec(select(ConversationSummary).limit(1))
            print("✅ Connexion à la base de données OK")
        except Exception as e:
            print(f"⚠️ Problème de connexion à la base de données: {e}")
            print("📝 Mode fallback activé - fonctionnalités limitées")
            self._fallback_mode = True
            self.enable_checkpointer = False
    
    def get_or_create_graph(self, character_name: str) -> StateGraph:
        """
        Récupère ou crée un graphe pour un personnage donné.
        
        Args:
            character_name: Nom du personnage
            
        Returns:
            Graphe compilé avec ou sans mémoire persistante
        """
        if character_name not in self._character_graphs:
            try:
                self._character_graphs[character_name] = create_character_graph_with_memory(
                    character_name=character_name,
                    enable_checkpointer=self.enable_checkpointer and not self._fallback_mode
                )
            except Exception as e:
                print(f"⚠️ Erreur création graphe pour {character_name}: {e}")
                print("📝 Création d'un graphe simplifié sans persistance")
                self._character_graphs[character_name] = self._create_simple_graph()
        
        return self._character_graphs[character_name]
    
    def _create_simple_graph(self) -> StateGraph:
        """Crée un graphe simplifié sans persistance pour les cas d'urgence."""
        graph = StateGraph(CharacterState)
        graph.add_node("simple_process", generate_simple_response)
        graph.set_entry_point("simple_process")
        graph.add_edge("simple_process", END)
        return graph.compile()
    
    async def process_message(
        self, 
        user_message: str, 
        character_data: dict,
        player_data: dict,
        thread_id: str = "default",
        session_id: Optional[str] = None
    ) -> dict:
        """
        Traite un message avec persistance automatique de la mémoire.
        🆕 Version améliorée avec support session_id complet.
        
        Args:
            user_message: Message de l'utilisateur
            character_data: Données du personnage
            thread_id: ID pour la persistance de conversation
            session_id: ID de session utilisateur (🆕 obligatoire pour filtering)
            
        Returns:
            État final avec la réponse générée
        """
        character_name = character_data.get("name", "unknown")
        
        # Récupération du graphe avec mémoire
        graph = self.get_or_create_graph(character_name)
        
        # 🆕 Construction de l'état initial avec session_id
        initial_state = self._build_initial_state(
            user_message, 
            character_data, 
            player_data,
            thread_id, 
            session_id
        )
        
        # 🆕 Configuration pour LangGraph avec thread_id ET session_id
        config = {
            "configurable": {
                "thread_id": thread_id,
                "session_id": session_id,
                "character_name": character_name
            }
        }
        
        # Exécution avec gestion d'erreurs robuste
        try:
            if self._fallback_mode:
                # Mode fallback : exécution simple sans persistance
                result = await self._execute_simple_fallback(initial_state, config)
            else:
                # Exécution normale avec persistance
                result = await graph.ainvoke(initial_state, config=config)
            
            # 🆕 Ajout d'informations sur la persistance et la mémoire avec session
            result["memory_info"] = {
                "thread_id": thread_id,
                "session_id": session_id,
                "character_name": character_name,
                "persistence_enabled": not self._fallback_mode,
                "checkpointer_enabled": self.enable_checkpointer,
                "context_summary_available": bool(result.get("context_summary")),
                "previous_summaries_count": len(result.get("previous_summaries", [])),
                "total_interactions": result.get("total_interactions", 0),
                "session_filtered": session_id is not None  # 🆕 Indique si filtré par session
            }
            
            # 🆕 Ajout des statistiques de mémoire pour l'interface avec session
            result["memory_stats"] = {
                "total_messages": result.get("total_interactions", 0),
                "summaries": len(result.get("previous_summaries", [])),
                "checkpoints": 0,  # À implémenter si nécessaire
                "last_activity": time.strftime("%H:%M:%S"),
                "context_loaded": bool(result.get("context_summary")),
                "memory_integration_enabled": result.get("memory_integration", {}).get("should_integrate", False),
                "session_id": session_id,  # 🆕 Inclut session_id dans les stats
                "session_specific": session_id is not None  # 🆕 Indique si données spécifiques à session
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erreur traitement message avec mémoire: {e}")
            # Fallback final
            return await self._execute_emergency_fallback(initial_state, config, str(e))
    
    async def _execute_simple_fallback(self, initial_state: CharacterState, config: dict) -> dict:
        """Exécute un traitement simple sans persistance."""
        try:
            # Traitement simplifié
            simple_graph = self._create_simple_graph()
            result = await simple_graph.ainvoke(initial_state, config=config)
            
            # Ajout d'informations de fallback
            result["fallback_info"] = {
                "reason": "database_unavailable",
                "mode": "simple_processing"
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erreur dans le fallback simple: {e}")
            return await self._execute_emergency_fallback(initial_state, config, str(e))
    
    async def _execute_emergency_fallback(self, initial_state: CharacterState, config: dict, error_msg: str) -> dict:
        """Fallback d'urgence avec réponse générée localement."""
        character_name = initial_state.get("character_name", "Personnage")
        user_message = initial_state.get("user_message", "")
        session_id = initial_state.get("session_id")
        
        # Réponse d'urgence
        emergency_response = f"Je suis {character_name}. Désolé, je rencontre des difficultés techniques en ce moment. Pouvez-vous répéter votre message ?"
        
        # État de retour minimal
        return {
            "response": emergency_response,
            "user_message": user_message,
            "character_name": character_name,
            "session_id": session_id,  # 🆕 Préserve session_id
            "emergency_fallback": True,
            "error_info": {
                "error": error_msg,
                "fallback_reason": "system_failure"
            },
            "conversation_history": initial_state.get("conversation_history", []),
            "debug_info": {
                "emergency_mode": True,
                "original_error": error_msg
            },
            "memory_stats": {
                "total_messages": 0,
                "summaries": 0,
                "checkpoints": 0,
                "last_activity": time.strftime("%H:%M:%S"),
                "context_loaded": False,
                "memory_integration_enabled": False,
                "session_id": session_id,  # 🆕 Inclut session_id même en fallback
                "session_specific": False
            }
        }
    
    def _build_initial_state(
        self, 
        user_message: str, 
        character_data: dict, 
        player_data: dict,
        thread_id: str,
        session_id: Optional[str]
    ) -> CharacterState:
        """
        Construit l'état initial avec informations de session.
        🆕 Version mise à jour avec session_id dans tous les champs pertinents.
        """
        
        return CharacterState(
            # Input
            user_message=user_message,
            response="",
            
            # Analyse (sera remplie par le graphe)
            parsed_message=None,
            message_intent=None,
            
            # Player
            player_data=player_data,

            # Personnage
            character_name=character_data.get("name", "unknown"),
            character_data=character_data,
            
            # Conversation avec persistance
            conversation_history=character_data.get("conversation_history", []),
            context_summary=None,  # Sera rempli par load_memory_context
            
            # 🆕 Mémoire persistante (sera remplie par load_memory_context avec session_id)
            previous_summaries=None,  # Sera rempli par load_memory_context
            memory_context=None,      # Sera rempli par load_memory_context
            total_interactions=None,  # Sera rempli par load_memory_context
            
            # 🆕 Informations de session (IMPORTANT!)
            thread_id=thread_id,
            session_id=session_id,  # 🆕 Session ID pour filtrage
            
            # RAG
            needs_rag_search=False,
            rag_query=None,
            rag_results=[],
            relevant_knowledge=[],
            needs_rag_retry=False,
            rag_retry_reason=None,
            
            # Actions
            input_trigger_probs=None,
            activated_input_triggers=None,
            refused_input_triggers=None,
            output_trigger_probs=None,
            
            # Déclencheurs de mémoire
            memory_trigger_activated=None,
            memory_trigger_type=None,
            memory_summary_created=None,
            
            # Métadonnées
            processing_start_time=0.0,
            processing_steps=[],
            debug_info={}
        )
    
    def get_conversation_history_summary(
        self, 
        character_name: str, 
        thread_id: str,
        session_id: Optional[str] = None,  # 🆕 Ajout session_id
        max_summaries: int = 5
    ) -> Dict[str, Any]:
        """
        Récupère un résumé de l'historique de conversation.
        🆕 Version mise à jour avec filtrage par session_id.
        
        Args:
            character_name: Nom du personnage
            thread_id: ID du thread
            session_id: ID de session pour filtrage (🆕)
            max_summaries: Nombre maximum de résumés à inclure
            
        Returns:
            Résumé structuré de l'historique
        """
        if self._fallback_mode:
            return {
                "summaries": [],
                "recent_messages": [],
                "total_interactions": 0,
                "session_id": session_id,  # 🆕 Inclut session_id
                "error": "Database unavailable"
            }
        
        try:
            from echoforge.agents.nodes.memory_update import EchoForgeMemoryManager
            
            llm_manager = LLMManager()
            memory_manager = EchoForgeMemoryManager(llm_manager)
            
            # 🆕 Appel avec session_id
            return memory_manager.get_conversation_context(
                character_name=character_name,
                thread_id=thread_id,
                session_id=session_id,  # 🆕 Filtrage par session
                include_summaries=True,
                max_summaries=max_summaries
            )
        except Exception as e:
            print(f"⚠️ Erreur récupération historique: {e}")
            return {
                "summaries": [],
                "recent_messages": [],
                "total_interactions": 0,
                "session_id": session_id,  # 🆕 Inclut session_id même en cas d'erreur
                "error": str(e)
            }
    
    def clear_conversation_memory(
        self, 
        character_name: str, 
        thread_id: str,
        session_id: Optional[str] = None,  # 🆕 Ajout session_id
        keep_summaries: bool = True
    ) -> bool:
        """
        Efface la mémoire de conversation pour un thread donné.
        🆕 Version mise à jour avec filtrage par session_id.
        
        Args:
            character_name: Nom du personnage
            thread_id: ID du thread
            session_id: ID de session pour filtrage (🆕)
            keep_summaries: Garder les résumés historiques
            
        Returns:
            True si succès, False sinon
        """
        if self._fallback_mode:
            print("⚠️ Effacement mémoire impossible - mode fallback")
            return False
        
        try:
            with get_session() as session:
                if not keep_summaries:
                    # 🆕 Suppression complète avec filtrage par session
                    stmt = select(ConversationSummary).where(
                        and_(
                            ConversationSummary.character_name == character_name,
                            ConversationSummary.thread_id == thread_id,
                            ConversationSummary.session_id == session_id if session_id else True
                        )
                    )
                    summaries = session.exec(stmt).all()
                    for summary in summaries:
                        session.delete(summary)
                
                session.commit()
                print(f"✅ Mémoire effacée pour {character_name} (session: {session_id})")
                return True
                
        except Exception as e:
            print(f"⚠️ Erreur effacement mémoire: {e}")
            return False
    
    def get_status(self) -> dict:
        """Retourne le statut du gestionnaire."""
        return {
            "database_available": not self._fallback_mode,
            "checkpointer_enabled": self.enable_checkpointer,
            "graphs_created": len(self._character_graphs),
            "fallback_mode": self._fallback_mode,
            "memory_context_loading": True,  # Indique que le chargement de contexte est actif
            "session_support": True  # 🆕 Indique le support des sessions
        }
    
    def toggle_checkpointer(self, enable: bool):
        """Active/désactive le checkpointer."""
        self.enable_checkpointer = enable
        # Efface les graphes existants pour qu'ils soient recréés
        self._character_graphs.clear()
        print(f"🔄 Checkpointer {'activé' if enable else 'désactivé'} - graphes rechargés")
    
    # 🆕 Méthodes utilitaires pour les sessions
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Récupère les statistiques pour une session donnée.
        
        Args:
            session_id: ID de la session
            
        Returns:
            Statistiques de la session
        """
        if self._fallback_mode:
            return {"error": "Database unavailable"}
        
        try:
            stats = {
                "session_id": session_id,
                "characters": {},
                "total_summaries": 0,
                "total_messages": 0
            }
            
            with get_session() as session:
                # Pour chaque personnage, récupère les stats
                for character_name in self._character_graphs.keys():
                    context = self.get_conversation_history_summary(
                        character_name=character_name,
                        thread_id=f"game_conversation_{character_name}",
                        session_id=session_id
                    )
                    
                    stats["characters"][character_name] = {
                        "summaries": len(context.get("summaries", [])),
                        "interactions": context.get("total_interactions", 0)
                    }
                    
                    stats["total_summaries"] += len(context.get("summaries", []))
                    stats["total_messages"] += context.get("total_interactions", 0)
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_sessions_with_memory(self) -> List[str]:
        """
        Liste les sessions ayant des données de mémoire.
        
        Returns:
            Liste des session_ids avec données
        """
        if self._fallback_mode:
            return []
        
        try:
            with get_session() as session:
                stmt = select(ConversationSummary.session_id).distinct()
                results = session.exec(stmt).all()
                return [result for result in results if result]
        except Exception as e:
            print(f"⚠️ Erreur récupération sessions: {e}")
            return []