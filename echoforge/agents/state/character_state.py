"""États typés pour les graphes LangGraph des personnages EchoForge."""

from typing import List, Dict, Optional, TypedDict, Any
from enum import Enum
from echoforge.utils.config import get_config

config = get_config()

class CharacterState(TypedDict):
    """État principal d'un personnage dans le graphe LangGraph."""
    
    # === INPUT/OUTPUT ===
    user_message: str
    response: str
    
    # === ANALYSE DU MESSAGE ===
    parsed_message: Optional[str]
    message_intent: Optional[str]
    
    # === CONTEXTE JEUX ===
    player_data: Dict[str, Any]

    # === CONTEXTE PERSONNAGE ===
    character_name: str
    character_data: Dict[str, Any]

    # === 🆕 IDENTIFIANTS DE SESSION ===
    thread_id: Optional[str] 
    session_id: Optional[str]  
    
    # === CONTEXTE CONVERSATIONNEL ===
    conversation_history: List[Dict[str, str]]
    context_summary: Optional[str]
    
    # === 🆕 MÉMOIRE PERSISTANTE ===
    previous_summaries: Optional[List[Dict[str, Any]]]  
    memory_context: Optional[Dict[str, Any]] 
    total_interactions: Optional[int] 

    # === RAG ET CONNAISSANCES ===
    needs_rag_search: bool
    rag_query: Optional[List[str]]
    rag_results: List[Any]
    relevant_knowledge: List[Any]
    needs_rag_retry: bool
    rag_retry_reason: Optional[str]
    
    # === ACTIONS ET DÉCLENCHEURS ===
    input_trigger_probs: Optional[Dict[str, float]]
    activated_input_triggers: Optional[List[str]]
    refused_input_triggers: Optional[List[str]]
    output_trigger_probs: Optional[Dict[str, Dict[str, Any]]]

    # === 🆕 DÉCLENCHEURS DE MÉMOIRE ===
    memory_trigger_activated: Optional[bool] 
    memory_trigger_type: Optional[str] 
    memory_summary_created: Optional[bool] 

    # === MÉTADONNÉES ===
    processing_start_time: float
    processing_steps: List[str]
    debug_info: Dict[str, Any]


# class ConversationState(TypedDict):
#     """État spécifique aux conversations."""
    
#     messages: List[Dict[str, str]]
#     current_turn: int
#     conversation_id: str
#     participant_emotions: Dict[str, str]


# class WorldState(TypedDict):
#     """État du monde du jeu."""
    
#     current_location: str
#     time_of_day: str
#     weather: str
#     active_events: List[str]
#     character_locations: Dict[str, str]
#     global_flags: Dict[str, bool]