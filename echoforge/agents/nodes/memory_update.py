import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.memory import ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from sqlmodel import Session, select, and_

from ..state.character_state import CharacterState
from echoforge.db.database import get_session
from echoforge.db.models.memory import ConversationSummary, ConversationMessage
from echoforge.core.llm_providers import LLMManager
from echoforge.utils.config import get_config
from langsmith import traceable

config = get_config()


class EchoForgeMemoryManager:
    """Gestionnaire de mémoire avancé pour EchoForge avec support des sessions."""
    
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
        trigger_info: Dict[str, Any],
        session_id: Optional[str] = None  # 🆕 Ajout du session_id
    ) -> Dict[str, Any]:
        """
        Crée un résumé de conversation en utilisant LangChain.
        🆕 Version mise à jour avec support session_id.
        
        Args:
            conversation_history: Historique des messages
            character_name: Nom du personnage
            thread_id: ID du thread
            trigger_info: Informations sur le déclencheur
            session_id: ID de la session (🆕)
            
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
            
            # Création du résumé avec contexte personnage ET session
            summary_prompt = f"""
Crée un résumé concis et informatif de cette conversation avec {character_name}.

Session: {session_id or 'Session par défaut'}
Thread: {thread_id}

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
            
            # 🆕 Métadonnées du résumé avec session_id
            summary_metadata = {
                "character_name": character_name,
                "session_id": session_id,  # 🆕 Ajout du session_id
                "thread_id": thread_id,
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
        session_id: Optional[str] = None  # 🆕 Ajout du session_id
    ) -> Optional[int]:
        """
        Sauvegarde le résumé en base de données.
        🆕 Version mise à jour avec support session_id.
        
        Returns:
            ID du résumé créé ou None en cas d'erreur
        """
        try:
            with get_session() as session:
                # Création du résumé avec session_id
                summary = ConversationSummary(
                    character_name=character_name,
                    thread_id=thread_id,
                    session_id=session_id,  # 🆕 Ajout du session_id
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
        session_id: Optional[str] = None  # 🆕 Ajout du session_id
    ):
        """
        Sauvegarde les messages en base avant résumé.
        🆕 Version mise à jour avec support session_id.
        """
        try:
            with get_session() as session:
                for i, msg in enumerate(conversation_history):
                    if isinstance(msg, dict):
                        # Message utilisateur
                        if "user" in msg:
                            user_msg = ConversationMessage(
                                character_name=character_name,
                                thread_id=thread_id,
                                session_id=session_id,  # 🆕 Ajout du session_id
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
                                session_id=session_id,  # 🆕 Ajout du session_id
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
        session_id: Optional[str] = None,  # 🆕 Ajout du session_id
        include_summaries: bool = True,
        max_summaries: int = 5
    ) -> Dict[str, Any]:
        """
        Récupère le contexte complet d'une conversation (résumés + messages récents).
        🆕 Version mise à jour avec filtrage par session_id.
        
        Args:
            character_name: Nom du personnage
            thread_id: ID du thread
            session_id: ID de la session pour filtrage (🆕)
            include_summaries: Inclure les résumés
            max_summaries: Nombre max de résumés
            
        Returns:
            Dict avec résumés et messages récents filtrés par session
        """
        context = {
            "summaries": [],
            "recent_messages": [],
            "total_interactions": 0,
            "session_id": session_id  # 🆕 Inclus dans le retour
        }
        
        try:
            with get_session() as session:
                if include_summaries:
                    # 🆕 Récupération des résumés filtrés par session_id
                    summaries_stmt = (
                        select(ConversationSummary)
                        .where(
                            and_(
                                ConversationSummary.character_name == character_name,
                                ConversationSummary.thread_id == thread_id,
                                # 🆕 Filtrage par session_id si fourni
                                ConversationSummary.session_id == session_id if session_id else True
                            )
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
                            "trigger_type": summary.trigger_type,
                            "session_id": summary.session_id  # 🆕 Inclus dans le retour
                        }
                        for summary in summaries
                    ]
                
                # 🆕 Comptage total des interactions filtrées par session
                total_messages_stmt = (
                    select(ConversationMessage)
                    .where(
                        and_(
                            ConversationMessage.character_name == character_name,
                            ConversationMessage.thread_id == thread_id,
                            # 🆕 Filtrage par session_id si fourni
                            ConversationMessage.session_id == session_id if session_id else True
                        )
                    )
                )
                
                total_messages = session.exec(total_messages_stmt).all()
                context["total_interactions"] = len(total_messages)
                
                print(f"🔍 Contexte récupéré pour session {session_id}: {len(context['summaries'])} résumés, {context['total_interactions']} messages")
                
        except Exception as e:
            print(f"⚠️ Erreur récupération contexte: {e}")
        
        return context


# 🆕 Nœud de chargement de mémoire mis à jour
@traceable
def load_memory_context(state: CharacterState) -> CharacterState:
    """
    🆕 Nœud pour charger le contexte de mémoire au début de la conversation.
    Version mise à jour avec filtrage par session_id.
    """
    state["processing_steps"].append("load_memory_context")
    
    character_name = state["character_name"]
    thread_id = state.get("thread_id", "default")
    session_id = state.get("session_id")  # 🆕 Récupération du session_id
    
    # Initialisation du gestionnaire de mémoire
    llm_manager = LLMManager()
    memory_manager = EchoForgeMemoryManager(llm_manager)
    
    try:
        # 🆕 Récupération du contexte de conversation filtré par session
        conversation_context = memory_manager.get_conversation_context(
            character_name=character_name,
            thread_id=thread_id,
            session_id=session_id,  # 🆕 Filtrage par session
            include_summaries=True,
            max_summaries=3
        )
        
        # Mise à jour de l'état avec le contexte récupéré
        state["previous_summaries"] = conversation_context.get("summaries", [])
        state["total_interactions"] = conversation_context.get("total_interactions", 0)
        
        # Création d'un résumé contextuel condensé
        if conversation_context.get("summaries"):
            # Combine les résumés en un contexte utilisable
            context_summary = _build_context_summary(conversation_context["summaries"])
            state["context_summary"] = context_summary
        else:
            state["context_summary"] = None
        
        # 🆕 Mise à jour des métadonnées de mémoire avec session_id
        state["memory_context"] = {
            "summaries_count": len(conversation_context.get("summaries", [])),
            "total_interactions": conversation_context.get("total_interactions", 0),
            "last_summary_date": conversation_context.get("summaries", [{}])[0].get("created_at") if conversation_context.get("summaries") else None,
            "memory_loaded": True,
            "session_id": session_id,  # 🆕 Inclus dans les métadonnées
            "filtered_by_session": session_id is not None  # 🆕 Indique si filtré
        }
        
        # Informations de debug
        state["debug_info"]["memory_context_loading"] = {
            "summaries_loaded": len(conversation_context.get("summaries", [])),
            "total_interactions": conversation_context.get("total_interactions", 0),
            "context_summary_created": bool(state["context_summary"]),
            "memory_available": True,
            "session_id": session_id,  # 🆕 Debug info avec session
            "session_filtering": session_id is not None
        }
        
        print(f"🧠 Contexte de mémoire chargé pour {character_name} (session: {session_id}): "
              f"{len(conversation_context.get('summaries', []))} résumés, "
              f"{conversation_context.get('total_interactions', 0)} interactions totales")
        
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du contexte de mémoire: {e}")
        
        # Valeurs par défaut en cas d'erreur
        state["previous_summaries"] = []
        state["total_interactions"] = 0
        state["context_summary"] = None
        state["memory_context"] = {
            "summaries_count": 0,
            "total_interactions": 0,
            "last_summary_date": None,
            "memory_loaded": False,
            "session_id": session_id,
            "filtered_by_session": False,
            "error": str(e)
        }
        
        state["debug_info"]["memory_context_loading"] = {
            "summaries_loaded": 0,
            "total_interactions": 0,
            "context_summary_created": False,
            "memory_available": False,
            "session_id": session_id,
            "session_filtering": False,
            "error": str(e)
        }
    
    return state


def _build_context_summary(summaries: List[Dict[str, Any]]) -> str:
    """
    Construit un résumé contextuel condensé à partir des résumés précédents.
    
    Args:
        summaries: Liste des résumés précédents
        
    Returns:
        Résumé contextuel condensé
    """
    if not summaries:
        return None
    
    # Trie les résumés par date (plus récent en premier)
    sorted_summaries = sorted(summaries, key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Prend les 3 résumés les plus récents
    recent_summaries = sorted_summaries[:3]
    
    context_parts = []
    
    for i, summary in enumerate(recent_summaries):
        summary_text = summary.get("text", "")
        messages_count = summary.get("messages_count", 0)
        session_id = summary.get("session_id", "")
        
        # Ajoute une indication temporelle
        if i == 0:
            prefix = "Récemment"
        elif i == 1:
            prefix = "Plus tôt"
        else:
            prefix = "Auparavant"
        
        # 🆕 Inclut l'info de session si disponible
        session_info = f" [Session: {session_id}]" if session_id else ""
        context_parts.append(f"{prefix} ({messages_count} échanges{session_info}): {summary_text}")
    
    # Combine tous les résumés
    full_context = "\n\n".join(context_parts)
    
    # Limite la longueur si nécessaire
    if len(full_context) > 1000:
        full_context = full_context[:1000] + "..."
    
    return full_context


@traceable
def check_memory_integration(state: CharacterState) -> CharacterState:
    """
    Nœud pour vérifier et intégrer la mémoire dans le processus de réponse.
    🆕 Version mise à jour avec informations de session.
    """
    state["processing_steps"].append("check_memory_integration")
    
    # Vérifie si on a un contexte de mémoire disponible
    has_memory_context = bool(state.get("context_summary"))
    has_previous_summaries = bool(state.get("previous_summaries"))
    session_id = state.get("session_id")
    
    # Détermine si la mémoire doit être intégrée dans la réponse
    should_integrate_memory = has_memory_context or has_previous_summaries
    
    # 🆕 Met à jour l'état avec les informations de mémoire et session
    state["memory_integration"] = {
        "has_context": has_memory_context,
        "has_summaries": has_previous_summaries,
        "should_integrate": should_integrate_memory,
        "summaries_count": len(state.get("previous_summaries", [])),
        "total_interactions": state.get("total_interactions", 0),
        "session_id": session_id,  # 🆕 Inclut le session_id
        "session_specific": session_id is not None  # 🆕 Indique si spécifique à une session
    }
    
    # Informations de debug
    state["debug_info"]["memory_integration"] = {
        "context_available": has_memory_context,
        "summaries_available": has_previous_summaries,
        "integration_enabled": should_integrate_memory,
        "context_length": len(state.get("context_summary", "") or ""),
        "summaries_count": len(state.get("previous_summaries", [])),
        "session_id": session_id,
        "session_filtering": session_id is not None
    }
    
    return state


# Nœud mis à jour pour la mise à jour de mémoire
@traceable
def update_character_memory(state: CharacterState) -> CharacterState:
    """
    Nœud de mise à jour de la mémoire avec le nouveau système et support session.
    🆕 Version mise à jour avec support session_id.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la mémoire gérée
    """
    state["processing_steps"].append("memory_update")
    
    # Initialisation du gestionnaire de mémoire
    llm_manager = LLMManager()
    memory_manager = EchoForgeMemoryManager(llm_manager)
    
    # 🆕 Récupération du session_id
    session_id = state.get("session_id")
    
    # Ajout du message actuel à l'historique
    current_exchange = {
        "user": state["user_message"],
        "assistant": state["response"],
        "timestamp": time.time(),
        "metadata": {
            "intent": state["message_intent"],
            "used_rag": bool(state["rag_results"]),
            "character_emotion": state["character_data"].get("current_emotion", "neutral"),
            "session_id": session_id  # 🆕 Inclut session_id dans metadata
        }
    }
    
    state["conversation_history"].append(current_exchange)
    
    # Vérification si un résumé est nécessaire
    summary_decision = memory_manager.should_create_summary(state)
    
    if summary_decision["should_summarize"]:
        print(f"🧠 Création d'un résumé (session: {session_id}): {summary_decision['reason']}")
        
        # 🆕 Sauvegarde des messages avant résumé avec session_id
        memory_manager.save_messages_to_db(
            conversation_history=state["conversation_history"],
            character_name=state["character_name"],
            thread_id=state.get("thread_id", "default"),
            session_id=session_id  # 🆕 Ajout du session_id
        )
        
        # 🆕 Création du résumé avec session_id
        summary_data = memory_manager.create_conversation_summary(
            conversation_history=state["conversation_history"],
            character_name=state["character_name"],
            thread_id=state.get("thread_id", "default"),
            trigger_info=summary_decision,
            session_id=session_id  # 🆕 Ajout du session_id
        )
        
        if summary_data["success"]:
            # 🆕 Sauvegarde du résumé en DB avec session_id
            summary_id = memory_manager.save_summary_to_db(
                summary_data=summary_data,
                character_name=state["character_name"],
                thread_id=state.get("thread_id", "default"),
                trigger_info=summary_decision,
                session_id=session_id  # 🆕 Ajout du session_id
            )
            
            # Troncature de l'historique
            state["conversation_history"] = memory_manager.truncate_conversation_history(
                state["conversation_history"]
            )
            
            # 🆕 Mise à jour du state avec les informations de résumé et session
            state["debug_info"]["memory_summary"] = {
                "summary_created": True,
                "summary_id": summary_id,
                "trigger_type": summary_decision["trigger_type"],
                "messages_summarized": summary_data["messages_count"],
                "messages_kept": len(state["conversation_history"]),
                "summary_text_preview": summary_data["summary_text"][:100] + "...",
                "session_id": session_id  # 🆕 Inclut session_id dans debug
            }
            
            print(f"✅ Résumé créé (ID: {summary_id}, Session: {session_id}) - {summary_data['messages_count']} messages résumés")
        else:
            print(f"❌ Erreur création résumé: {summary_data.get('error', 'Inconnue')}")
            state["debug_info"]["memory_summary"] = {
                "summary_created": False,
                "error": summary_data.get("error", "Erreur inconnue"),
                "session_id": session_id
            }
    else:
        # Pas de résumé nécessaire
        state["debug_info"]["memory_update"] = {
            "summary_created": False,
            "reason": summary_decision["reason"],
            "conversation_length": len(state["conversation_history"]),
            "session_id": session_id  # 🆕 Inclut session_id dans debug
        }
    
    return state


@traceable
def finalize_interaction(state: CharacterState) -> CharacterState:
    """
    Finalise l'interaction avec les nouvelles métriques mémoire.
    🆕 Version mise à jour avec informations de session.
    
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
    
    # 🆕 Récupération du contexte avec session_id
    session_id = state.get("session_id")
    conversation_context = memory_manager.get_conversation_context(
        character_name=state["character_name"],
        thread_id=state.get("thread_id", "default"),
        session_id=session_id  # 🆕 Filtrage par session
    )
    
    # 🆕 Mise à jour des métadonnées finales avec informations de session
    state["debug_info"]["final_stats"] = {
        "total_processing_time": processing_time,
        "steps_count": len(state["processing_steps"]),
        "rag_used": bool(state["rag_results"]),
        "response_length": len(state["response"]),
        "session_id": session_id,  # 🆕 Inclut session_id
        "conversation_stats": {
            "current_messages": len(state["conversation_history"]),
            "total_summaries": len(conversation_context["summaries"]),
            "total_interactions": conversation_context["total_interactions"],
            "session_specific": session_id is not None  # 🆕 Indique si données spécifiques à session
        }
    }
    
    return state