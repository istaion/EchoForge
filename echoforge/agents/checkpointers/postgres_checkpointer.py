from typing import Any, Dict, Optional, Sequence, Tuple, List
from uuid import uuid4
import json
from datetime import datetime
import asyncio
import uuid
import binascii

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from sqlmodel import Session, select, and_
from echoforge.db.database import get_session
from echoforge.db.models.memory import ConversationSummary


class PostgreSQLCheckpointSaver(BaseCheckpointSaver):
    """
    Checkpointer PostgreSQL intégré avec le système de mémoire EchoForge.
    Version corrigée avec support UUID approprié.
    """
    
    def __init__(self, character_name: str):
        super().__init__()
        self.character_name = character_name
        self.enabled = True  # 🆕 Permet de désactiver le checkpointer
    
    def _generate_checkpoint_id(self) -> str:
        """Génère un ID de checkpoint compatible avec LangGraph (UUID hex)."""
        return str(uuid.uuid4())
    
    def _db_id_to_checkpoint_id(self, db_id: int) -> str:
        """Convertit un ID de base de données en checkpoint ID compatible."""
        # Crée un UUID déterministe basé sur l'ID de base de données
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace fixe
        return str(uuid.uuid5(namespace, f"{self.character_name}_{db_id}"))
    
    def _checkpoint_id_to_db_id(self, checkpoint_id: str) -> Optional[int]:
        """Essaie de récupérer l'ID de base de données à partir d'un checkpoint ID."""
        try:
            with get_session() as session:
                # Recherche par l'ID de checkpoint généré
                stmt = select(ConversationSummary).where(
                    and_(
                        ConversationSummary.character_name == self.character_name,
                        ConversationSummary.summary_metadata.op('->>')('checkpoint_id') == checkpoint_id
                    )
                )
                summary = session.exec(stmt).first()
                return summary.id if summary else None
        except Exception:
            return None
    
    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Récupère un checkpoint spécifique."""
        if not self.enabled:
            return None
            
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        if not thread_id:
            return None
        
        try:
            with get_session() as session:
                if checkpoint_id:
                    # Recherche par checkpoint_id spécifique
                    db_id = self._checkpoint_id_to_db_id(checkpoint_id)
                    if not db_id:
                        return None
                    
                    stmt = select(ConversationSummary).where(
                        ConversationSummary.id == db_id
                    )
                    summary = session.exec(stmt).first()
                else:
                    # Récupère le dernier checkpoint pour ce thread
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
                
                if not summary:
                    return None
                
                # Génère un checkpoint ID compatible
                checkpoint_id = self._db_id_to_checkpoint_id(summary.id)
                
                # Mise à jour des métadonnées avec le checkpoint_id
                summary_metadata = summary.summary_metadata.copy()
                summary_metadata['checkpoint_id'] = checkpoint_id
                
                checkpoint = Checkpoint(
                    v=1,
                    ts=summary.created_at.isoformat(),
                    id=checkpoint_id,
                    channel_values={
                        "conversation_summary": summary.summary_text,
                        "messages_count": summary.messages_count,
                        "character_name": self.character_name,
                        "thread_id": thread_id
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
                            "summary_text": summary.summary_text,
                            "db_id": summary.id
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
        if not self.enabled:
            return []
            
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
                    checkpoint_id = self._db_id_to_checkpoint_id(summary.id)
                    
                    # Mise à jour des métadonnées
                    summary_metadata = summary.summary_metadata.copy()
                    summary_metadata['checkpoint_id'] = checkpoint_id
                    
                    checkpoint = Checkpoint(
                        v=1,
                        ts=summary.created_at.isoformat(),
                        id=checkpoint_id,
                        channel_values={
                            "conversation_summary": summary.summary_text,
                            "messages_count": summary.messages_count,
                            "character_name": self.character_name,
                            "thread_id": thread_id
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
                                "summary_text": summary.summary_text,
                                "db_id": summary.id
                            }
                        }
                    )
                    
                    checkpoint_config = config.copy()
                    checkpoint_config["configurable"]["checkpoint_id"] = checkpoint_id
                    
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
        """Sauvegarde un checkpoint."""
        if not self.enabled:
            return
            
        # Les checkpoints sont automatiquement créés par le système de mémoire
        # Cette méthode pourrait être utilisée pour des sauvegardes manuelles
        pass

    # Méthodes async pour LangGraph
    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Version async de get_tuple."""
        if not self.enabled:
            return None
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: dict,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = 10
    ) -> Sequence[CheckpointTuple]:
        """Version async de list."""
        if not self.enabled:
            return []
        return await asyncio.to_thread(self.list, config, filter=filter, before=before, limit=limit)

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        checkpoint_ns: str = ""
    ) -> None:
        """Version async de put."""
        if not self.enabled:
            return
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, checkpoint_ns)
    
    def put_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """Sauvegarde les écritures intermédiaires."""
        if not self.enabled:
            return
        # Pour l'instant, on ne fait rien avec les écritures intermédiaires
        pass
    
    async def aput_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """Version async de put_writes."""
        if not self.enabled:
            return
        return await asyncio.to_thread(self.put_writes, config, writes, task_id)
    
    def list_session_ids(self) -> List[str]:
        """
        Récupère tous les `session_id` uniques présents dans la base pour ce personnage.
        """
        if not self.enabled:
            return []
        
        try:
            with get_session() as session:
                stmt = (
                    select(ConversationSummary.session_id)
                    .where(
                        and_(
                            ConversationSummary.character_name == self.character_name,
                            ConversationSummary.session_id.is_not(None)
                        )
                    )
                    .distinct()
                )
                rows = session.exec(stmt).all()
                return [row[0] for row in rows if row[0]]
        except Exception as e:
            print(f"⚠️ Erreur récupération session_ids: {e}")
        return []
    
    def disable(self):
        """Désactive le checkpointer."""
        self.enabled = False
        print("⚠️ Checkpointer PostgreSQL désactivé")
    
    def enable(self):
        """Réactive le checkpointer."""
        self.enabled = True
        print("✅ Checkpointer PostgreSQL réactivé")


# 🆕 Classe de fallback sans checkpointer
class NoOpCheckpointSaver(BaseCheckpointSaver):
    """Checkpointer vide pour les cas de fallback."""
    
    def __init__(self):
        super().__init__()
    
    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return None
    
    def list(self, config: dict, **kwargs) -> Sequence[CheckpointTuple]:
        return []
    
    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, checkpoint_ns: str = "") -> None:
        pass
    
    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return None
    
    async def alist(self, config: dict, **kwargs) -> Sequence[CheckpointTuple]:
        return []
    
    async def aput(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, checkpoint_ns: str = "") -> None:
        pass
    
    def put_writes(self, config: dict, writes: List[Tuple[str, Any]], task_id: str) -> None:
        pass
    
    async def aput_writes(self, config: dict, writes: List[Tuple[str, Any]], task_id: str) -> None:
        pass


def create_safe_checkpointer(character_name: str, enable_checkpointer: bool = True) -> BaseCheckpointSaver:
    """
    Crée un checkpointer sûr avec fallback automatique.
    
    Args:
        character_name: Nom du personnage
        enable_checkpointer: Si False, utilise un checkpointer vide
        
    Returns:
        Checkpointer fonctionnel ou fallback
    """
    if not enable_checkpointer:
        print("📝 Utilisation du checkpointer vide (NoOp)")
        return NoOpCheckpointSaver()
    
    try:
        # Teste la connexion à la base de données
        with get_session() as session:
            session.exec(select(ConversationSummary).limit(1))
        
        # Crée le checkpointer PostgreSQL
        checkpointer = PostgreSQLCheckpointSaver(character_name)
        print(f"✅ Checkpointer PostgreSQL créé pour {character_name}")
        return checkpointer
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création du checkpointer PostgreSQL: {e}")
        print("📝 Utilisation du checkpointer vide comme fallback")
        return NoOpCheckpointSaver()