"""États typés pour les graphes LangGraph des personnages EchoForge."""

from typing import List, Dict, Optional, TypedDict, Any
from enum import Enum


class ComplexityLevel(str, Enum):
    """Niveaux de complexité pour les requêtes."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class CharacterState(TypedDict):
    """État principal d'un personnage dans le graphe LangGraph."""
    
    # === INPUT/OUTPUT ===
    user_message: str
    response: str
    
    # === ANALYSE DU MESSAGE ===
    parsed_message: Optional[str]
    message_intent: Optional[str]
    complexity_level: ComplexityLevel
    
    # === CONTEXTE PERSONNAGE ===
    character_name: str
    personality_traits: Dict[str, Any]
    current_emotion: str
    character_knowledge: List[str]
    
    # === CONTEXTE CONVERSATIONNEL ===
    conversation_history: List[Dict[str, str]]
    context_summary: Optional[str]
    
    # === RAG ET CONNAISSANCES ===
    needs_rag_search: bool
    rag_query: Optional[str]
    rag_results: List[Dict[str, Any]]
    relevant_knowledge: List[str]
    
    # === ACTIONS ET DÉCLENCHEURS ===
    planned_actions: List[str]
    triggered_events: List[str]
    game_state_changes: Dict[str, Any]
    
    # === MÉTADONNÉES ===
    processing_start_time: float
    processing_steps: List[str]
    debug_info: Dict[str, Any]


class ConversationState(TypedDict):
    """État spécifique aux conversations."""
    
    messages: List[Dict[str, str]]
    current_turn: int
    conversation_id: str
    participant_emotions: Dict[str, str]


class WorldState(TypedDict):
    """État du monde du jeu."""
    
    current_location: str
    time_of_day: str
    weather: str
    active_events: List[str]
    character_locations: Dict[str, str]
    global_flags: Dict[str, bool]