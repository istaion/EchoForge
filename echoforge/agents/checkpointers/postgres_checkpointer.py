from typing import Any, Dict, Optional, Sequence, Tuple, List
from uuid import uuid4
import json
from datetime import datetime
import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from sqlmodel import Session, select, and_
from echoforge.db.database import get_session
from echoforge.db.models.memory import ConversationSummary


class PostgreSQLCheckpointSaver(BaseCheckpointSaver):
    """
    Checkpointer PostgreSQL intégré avec le système de mémoire EchoForge.
    """
    
    def __init__(self, character_name: str):
        super().__init__()
        self.character_name = character_name
    
    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Récupère un checkpoint spécifique."""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        if not thread_id:
            return None
        
        try:
            with get_session() as session:
                # Pour l'instant, on utilise les résumés comme checkpoints
                stmt = (
                    select(ConversationSummary)
                    .where(
                        and_(
                            ConversationSummary.character_name == self.character_name,
                            ConversationSummary.thread_id == thread_id
                        )
                    )
                    .order_by(ConversationSummary.created_at.desc())
                    .limit(1)
                )
                
                summary = session.exec(stmt).first()
                
                if summary:
                    checkpoint = Checkpoint(
                        v=1,
                        ts=summary.created_at.isoformat(),
                        id=str(summary.id),
                        channel_values={
                            "conversation_summary": summary.summary_text,
                            "messages_count": summary.messages_count,
                            "character_name": self.character_name
                        },
                        channel_versions={
                            "__root__": summary.id
                        },
                        versions_seen={}
                    )
                    
                    metadata = CheckpointMetadata(
                        source="database",
                        step=summary.messages_count,
                        writes={
                            "summary": {
                                "trigger_type": summary.trigger_type,
                                "summary_text": summary.summary_text
                            }
                        }
                    )
                    
                    return CheckpointTuple(
                        config=config,
                        checkpoint=checkpoint,
                        metadata=metadata
                    )
        
        except Exception as e:
            print(f"⚠️ Erreur récupération checkpoint: {e}")
        
        return None
    
    def list(
        self,
        config: dict,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = 10
    ) -> Sequence[CheckpointTuple]:
        """Liste les checkpoints disponibles."""
        thread_id = config.get("configurable", {}).get("thread_id")
        
        if not thread_id:
            return []
        
        checkpoints = []
        
        try:
            with get_session() as session:
                stmt = (
                    select(ConversationSummary)
                    .where(
                        and_(
                            ConversationSummary.character_name == self.character_name,
                            ConversationSummary.thread_id == thread_id
                        )
                    )
                    .order_by(ConversationSummary.created_at.desc())
                )
                
                if limit:
                    stmt = stmt.limit(limit)
                
                summaries = session.exec(stmt).all()
                
                for summary in summaries:
                    checkpoint = Checkpoint(
                        v=1,
                        ts=summary.created_at.isoformat(),
                        id=str(summary.id),
                        channel_values={
                            "conversation_summary": summary.summary_text,
                            "messages_count": summary.messages_count,
                            "character_name": self.character_name
                        },
                        channel_versions={
                            "__root__": summary.id
                        },
                        versions_seen={}
                    )
                    
                    metadata = CheckpointMetadata(
                        source="database",
                        step=summary.messages_count,
                        writes={
                            "summary": {
                                "trigger_type": summary.trigger_type,
                                "summary_text": summary.summary_text
                            }
                        }
                    )
                    
                    checkpoint_config = config.copy()
                    checkpoint_config["configurable"]["checkpoint_id"] = str(summary.id)
                    
                    checkpoints.append(CheckpointTuple(
                        config=checkpoint_config,
                        checkpoint=checkpoint,
                        metadata=metadata
                    ))
        
        except Exception as e:
            print(f"⚠️ Erreur listage checkpoints: {e}")
        
        return checkpoints
    
    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        checkpoint_ns: str = ""
    ) -> None:
        """Sauvegarde un checkpoint (géré par le système de mémoire)."""
        # Les checkpoints sont automatiquement créés par le système de mémoire
        # Cette méthode pourrait être utilisée pour des sauvegardes manuelles
        pass

    # Gestion pour les machin asyncrone
    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Version async de get_tuple pour LangGraph async."""
        # Simple adaptation si tu veux réutiliser ton code sync
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: dict,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = 10
    ) -> Sequence[CheckpointTuple]:
        return await asyncio.to_thread(self.list, config, filter=filter, before=before, limit=limit)

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        checkpoint_ns: str = ""
    ) -> None:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, checkpoint_ns)
    
    def put_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """Sauvegarde les écritures intermédiaires."""
        # Pour l'instant, on ne fait rien avec les écritures intermédiaires
        # car elles sont gérées par le système de mémoire EchoForge
        pass
    
    async def aput_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """Version async de put_writes."""
        return await asyncio.to_thread(self.put_writes, config, writes, task_id)