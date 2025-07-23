from sqlmodel import SQLModel, Field, Column
from sqlalchemy import DateTime, Text, JSON
from datetime import datetime
from typing import Optional, Dict, Any


class ConversationSummary(SQLModel, table=True):
    """Résumés de conversations stockés en base."""
    
    __tablename__ = "conversation_summaries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Identifiants
    character_name: str = Field(index=True, description="Nom du personnage")
    thread_id: str = Field(index=True, description="ID du thread de conversation")
    session_id: Optional[str] = Field(default=None, index=True, description="ID de session utilisateur")
    
    # Contenu du résumé
    summary_text: str = Field(sa_column=Column(Text), description="Texte du résumé")
    summary_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        sa_column=Column(JSON),
        description="Métadonnées du résumé"
    )
    
    # Informations sur la période résumée
    messages_count: int = Field(description="Nombre de messages résumés")
    start_timestamp: datetime = Field(description="Début de la période résumée")
    end_timestamp: datetime = Field(description="Fin de la période résumée")
    
    # Trigger qui a causé le résumé
    trigger_type: str = Field(description="Type de déclencheur: 'bye' ou 'auto'")
    trigger_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Métadonnées du déclencheur"
    )
    
    # Timestamps système
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True))
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )


class ConversationMessage(SQLModel, table=True):
    """Messages de conversation pour backup et analyse."""
    
    __tablename__ = "conversation_messages"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Identifiants
    character_name: str = Field(index=True)
    thread_id: str = Field(index=True)
    session_id: Optional[str] = Field(default=None, index=True)
    
    # Contenu du message
    role: str = Field(description="'user' ou 'assistant'")
    content: str = Field(sa_column=Column(Text))
    message_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    
    # Ordre dans la conversation
    sequence_number: int = Field(description="Numéro de séquence dans la conversation")
    
    # Statut
    is_summarized: bool = Field(default=False, description="Message inclus dans un résumé")
    summary_id: Optional[int] = Field(default=None, foreign_key="conversation_summaries.id")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True))
    )


class GameSession(SQLModel, table=True):
    """Sessions de jeu stockées en base."""
    
    __tablename__ = "game_sessions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Identifiants
    session_id: str = Field(index=True, unique=True, description="ID unique de la session")
    session_name: str = Field(description="Nom affiché de la session")
    
    # Données de jeu
    player_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Données complètes du joueur"
    )
    characters_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Données complètes des personnages"
    )
    
    # Métadonnées
    game_state: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="État du jeu (position montgolfière, etc.)"
    )
    
    # Statistiques
    total_playtime_seconds: int = Field(default=0, description="Temps de jeu total en secondes")
    last_character_talked: Optional[str] = Field(default=None, description="Dernier personnage parlé")
    messages_count: int = Field(default=0, description="Nombre total de messages")
    
    # Statut
    is_active: bool = Field(default=True, description="Session active ou archivée")
    is_completed: bool = Field(default=False, description="Jeu terminé (montgolfière réparée)")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True))
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    last_played_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )


class SessionEvent(SQLModel, table=True):
    """Événements de session pour analytics."""
    
    __tablename__ = "session_events"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    session_id: str = Field(index=True, foreign_key="game_sessions.session_id")
    event_type: str = Field(description="Type d'événement (conversation, quest, etc.)")
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True))
    )