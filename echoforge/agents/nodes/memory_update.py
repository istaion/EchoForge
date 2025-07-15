import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.memory import ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from sqlmodel import Session, select

from ..state.character_state import CharacterState
from echoforge.db.database import get_session
from echoforge.db.models.memory import ConversationSummary, ConversationMessage
from echoforge.core.llm_providers import LLMManager
from echoforge.utils.config import get_config
from langsmith import traceable

config = get_config()


class EchoForgeMemoryManager:
    """Gestionnaire de mémoire avancé pour EchoForge."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.config = get_config()
        
        # Configuration des seuils
        self.max_messages_without_summary = self.config.max_messages_without_summary
        self.keep_recent_messages = self.config.keep_recent_messages
        
        # Initialisation de ConversationSummaryMemory
        self.summary_memory = ConversationSummaryMemory(
            llm=llm_manager.get_llm(),
            return_messages=True,
            max_token_limit=2000
        )
    
    @traceable
    def should_create_summary(self, state: CharacterState) -> Dict[str, Any]:
        """
        Détermine si un résumé doit être créé.
        
        Returns:
            Dict avec 'should_summarize', 'trigger_type', et métadonnées
        """
        activated_triggers = state.get("activated_input_triggers", [])
        conversation_history = state.get("conversation_history", [])
        
        # Déclencheur 1: Action "bye" détectée
        if "bye" in activated_triggers:
            return {
                "should_summarize": True,
                "trigger_type": "bye",
                "trigger_metadata": {
                    "activated_triggers": activated_triggers,
                    "bye_detected": True
                },
                "reason": "Déclencheur 'bye' activé"
            }
        
        # Déclencheur 2: Trop de messages sans résumé
        if len(conversation_history) >= self.max_messages_without_summary:
            return {
                "should_summarize": True,
                "trigger_type": "auto",
                "trigger_metadata": {
                    "messages_count": len(conversation_history),
                    "threshold": self.max_messages_without_summary
                },
                "reason": f"Seuil de {self.max_messages_without_summary} messages atteint"
            }
        
        return {
            "should_summarize": False,
            "trigger_type": None,
            "trigger_metadata": {},
            "reason": "Aucun déclencheur activé"
        }
    
    @traceable
    def create_conversation_summary(
        self, 
        conversation_history: List[Dict[str, Any]],
        character_name: str,
        thread_id: str,
        trigger_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crée un résumé de conversation en utilisant LangChain.
        
        Args:
            conversation_history: Historique des messages
            character_name: Nom du personnage
            thread_id: ID du thread
            trigger_info: Informations sur le déclencheur
            
        Returns:
            Dict avec le résumé et métadonnées
        """
        try:
            # Conversion des messages pour LangChain
            messages = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    if "user" in msg and "assistant" in msg:
                        messages.append(HumanMessage(content=msg["user"]))
                        messages.append(AIMessage(content=msg["assistant"]))
                    elif msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            if not messages:
                return {"success": False, "error": "Aucun message à résumer"}
            
            # Création du résumé avec contexte personnage
            summary_prompt = f"""
Crée un résumé concis et informatif de cette conversation avec {character_name}.

Instructions:
1. Garde les informations importantes sur les relations et émotions
2. Mentionne les événements clés et décisions prises
3. Préserve le contexte narratif et les éléments de jeu
4. Utilise un style neutre et factuel
5. Maximum 200 mots

Conversation à résumer:
"""
            
            # Utilisation de ConversationSummaryMemory
            for message in messages:
                self.summary_memory.chat_memory.add_message(message)
            
            # Génération du résumé
            summary_text = self.summary_memory.predict_new_summary(
                messages=messages,
                existing_summary=""
            )
            
            # Métadonnées du résumé
            summary_metadata = {
                "character_name": character_name,
                "messages_processed": len(messages),
                "summary_length": len(summary_text),
                "creation_method": "langchain_summary_memory",
                "llm_model": self.config.llm_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return {
                "success": True,
                "summary_text": summary_text,
                "summary_metadata": summary_metadata,
                "messages_count": len(conversation_history),
                "start_timestamp": self._get_first_message_timestamp(conversation_history),
                "end_timestamp": self._get_last_message_timestamp(conversation_history)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_summary": f"Conversation avec {character_name} - {len(conversation_history)} échanges"
            }
    
    def _get_first_message_timestamp(self, history: List[Dict]) -> datetime:
        """Récupère le timestamp du premier message."""
        for msg in history:
            if "timestamp" in msg:
                if isinstance(msg["timestamp"], (int, float)):
                    return datetime.fromtimestamp(msg["timestamp"])
                elif isinstance(msg["timestamp"], str):
                    return datetime.fromisoformat(msg["timestamp"])
        return datetime.utcnow()
    
    def _get_last_message_timestamp(self, history: List[Dict]) -> datetime:
        """Récupère le timestamp du dernier message."""
        for msg in reversed(history):
            if "timestamp" in msg:
                if isinstance(msg["timestamp"], (int, float)):
                    return datetime.fromtimestamp(msg["timestamp"])
                elif isinstance(msg["timestamp"], str):
                    return datetime.fromisoformat(msg["timestamp"])
        return datetime.utcnow()
    
    @traceable
    def save_summary_to_db(
        self,
        summary_data: Dict[str, Any],
        character_name: str,
        thread_id: str,
        trigger_info: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Sauvegarde le résumé en base de données.
        
        Returns:
            ID du résumé créé ou None en cas d'erreur
        """
        try:
            with get_session() as session:
                # Création du résumé
                summary = ConversationSummary(
                    character_name=character_name,
                    thread_id=thread_id,
                    session_id=session_id,
                    summary_text=summary_data["summary_text"],
                    summary_metadata=summary_data["summary_metadata"],
                    messages_count=summary_data["messages_count"],
                    start_timestamp=summary_data["start_timestamp"],
                    end_timestamp=summary_data["end_timestamp"],
                    trigger_type=trigger_info["trigger_type"],
                    trigger_metadata=trigger_info["trigger_metadata"]
                )
                
                session.add(summary)
                session.commit()
                session.refresh(summary)
                
                return summary.id
                
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde résumé DB: {e}")
            return None
    
    @traceable
    def save_messages_to_db(
        self,
        conversation_history: List[Dict[str, Any]],
        character_name: str,
        thread_id: str,
        summary_id: Optional[int] = None,
        session_id: Optional[str] = None
    ):
        """Sauvegarde les messages en base avant résumé."""
        try:
            with get_session() as session:
                for i, msg in enumerate(conversation_history):
                    if isinstance(msg, dict):
                        # Message utilisateur
                        if "user" in msg:
                            user_msg = ConversationMessage(
                                character_name=character_name,
                                thread_id=thread_id,
                                session_id=session_id,
                                role="user",
                                content=msg["user"],
                                message_metadata=msg.get("metadata", {}),
                                sequence_number=i * 2,
                                is_summarized=summary_id is not None,
                                summary_id=summary_id
                            )
                            session.add(user_msg)
                        
                        # Message assistant
                        if "assistant" in msg:
                            assistant_msg = ConversationMessage(
                                character_name=character_name,
                                thread_id=thread_id,
                                session_id=session_id,
                                role="assistant", 
                                content=msg["assistant"],
                                message_metadata=msg.get("metadata", {}),
                                sequence_number=i * 2 + 1,
                                is_summarized=summary_id is not None,
                                summary_id=summary_id
                            )
                            session.add(assistant_msg)
                
                session.commit()
                
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde messages DB: {e}")
    
    @traceable
    def truncate_conversation_history(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Garde seulement les N derniers messages selon la configuration.
        
        Args:
            conversation_history: Historique complet
            
        Returns:
            Historique tronqué
        """
        if len(conversation_history) <= self.keep_recent_messages:
            return conversation_history
        
        return conversation_history[-self.keep_recent_messages:]
    
    @traceable
    def get_conversation_context(
        self,
        character_name: str,
        thread_id: str,
        include_summaries: bool = True,
        max_summaries: int = 5
    ) -> Dict[str, Any]:
        """
        Récupère le contexte complet d'une conversation (résumés + messages récents).
        
        Returns:
            Dict avec résumés et messages récents
        """
        context = {
            "summaries": [],
            "recent_messages": [],
            "total_interactions": 0
        }
        
        try:
            with get_session() as session:
                if include_summaries:
                    # Récupération des résumés récents
                    summaries_stmt = (
                        select(ConversationSummary)
                        .where(
                            ConversationSummary.character_name == character_name,
                            ConversationSummary.thread_id == thread_id
                        )
                        .order_by(ConversationSummary.created_at.desc())
                        .limit(max_summaries)
                    )
                    
                    summaries = session.exec(summaries_stmt).all()
                    context["summaries"] = [
                        {
                            "text": summary.summary_text,
                            "created_at": summary.created_at,
                            "messages_count": summary.messages_count,
                            "trigger_type": summary.trigger_type
                        }
                        for summary in summaries
                    ]
                
                # Comptage total des interactions
                total_messages_stmt = (
                    select(ConversationMessage)
                    .where(
                        ConversationMessage.character_name == character_name,
                        ConversationMessage.thread_id == thread_id
                    )
                )
                
                total_messages = session.exec(total_messages_stmt).all()
                context["total_interactions"] = len(total_messages)
                
        except Exception as e:
            print(f"⚠️ Erreur récupération contexte: {e}")
        
        return context


# Nœuds LangGraph mis à jour
@traceable
def update_character_memory(state: CharacterState) -> CharacterState:
    """
    Nœud de mise à jour de la mémoire avec le nouveau système.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la mémoire gérée
    """
    state["processing_steps"].append("memory_update")
    
    # Initialisation du gestionnaire de mémoire
    llm_manager = LLMManager()
    memory_manager = EchoForgeMemoryManager(llm_manager)
    
    # Ajout du message actuel à l'historique
    current_exchange = {
        "user": state["user_message"],
        "assistant": state["response"],
        "timestamp": time.time(),
        "metadata": {
            "intent": state["message_intent"],
            "used_rag": bool(state["rag_results"]),
            "character_emotion": state["character_data"].get("current_emotion", "neutral")
        }
    }
    
    state["conversation_history"].append(current_exchange)
    
    # Vérification si un résumé est nécessaire
    summary_decision = memory_manager.should_create_summary(state)
    
    if summary_decision["should_summarize"]:
        print(f"🧠 Création d'un résumé: {summary_decision['reason']}")
        
        # Sauvegarde des messages avant résumé
        memory_manager.save_messages_to_db(
            conversation_history=state["conversation_history"],
            character_name=state["character_name"],
            thread_id=state.get("thread_id", "default"),
            session_id=state.get("session_id")
        )
        
        # Création du résumé
        summary_data = memory_manager.create_conversation_summary(
            conversation_history=state["conversation_history"],
            character_name=state["character_name"],
            thread_id=state.get("thread_id", "default"),
            trigger_info=summary_decision
        )
        
        if summary_data["success"]:
            # Sauvegarde du résumé en DB
            summary_id = memory_manager.save_summary_to_db(
                summary_data=summary_data,
                character_name=state["character_name"],
                thread_id=state.get("thread_id", "default"),
                trigger_info=summary_decision,
                session_id=state.get("session_id")
            )
            
            # Troncature de l'historique
            state["conversation_history"] = memory_manager.truncate_conversation_history(
                state["conversation_history"]
            )
            
            # Mise à jour du state avec les informations de résumé
            state["debug_info"]["memory_summary"] = {
                "summary_created": True,
                "summary_id": summary_id,
                "trigger_type": summary_decision["trigger_type"],
                "messages_summarized": summary_data["messages_count"],
                "messages_kept": len(state["conversation_history"]),
                "summary_text_preview": summary_data["summary_text"][:100] + "..."
            }
            
            print(f"✅ Résumé créé (ID: {summary_id}) - {summary_data['messages_count']} messages résumés")
        else:
            print(f"❌ Erreur création résumé: {summary_data.get('error', 'Inconnue')}")
            state["debug_info"]["memory_summary"] = {
                "summary_created": False,
                "error": summary_data.get("error", "Erreur inconnue")
            }
    else:
        # Pas de résumé nécessaire
        state["debug_info"]["memory_update"] = {
            "summary_created": False,
            "reason": summary_decision["reason"],
            "conversation_length": len(state["conversation_history"])
        }
    
    return state


@traceable
def finalize_interaction(state: CharacterState) -> CharacterState:
    """
    Finalise l'interaction avec les nouvelles métriques mémoire.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État finalisé
    """
    state["processing_steps"].append("finalization")
    
    # Calcul du temps de traitement total
    processing_time = time.time() - state["processing_start_time"]
    
    # Récupération du contexte de conversation complet
    llm_manager = LLMManager()
    memory_manager = EchoForgeMemoryManager(llm_manager)
    
    conversation_context = memory_manager.get_conversation_context(
        character_name=state["character_name"],
        thread_id=state.get("thread_id", "default")
    )
    
    # Mise à jour des métadonnées finales
    state["debug_info"]["final_stats"] = {
        "total_processing_time": processing_time,
        "steps_count": len(state["processing_steps"]),
        "rag_used": bool(state["rag_results"]),
        "response_length": len(state["response"]),
        "conversation_stats": {
            "current_messages": len(state["conversation_history"]),
            "total_summaries": len(conversation_context["summaries"]),
            "total_interactions": conversation_context["total_interactions"]
        }
    }
    
    return state