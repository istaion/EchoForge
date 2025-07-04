"""Routeurs conditionnels pour les graphes LangGraph."""

from ..state.character_state import CharacterState, ComplexityLevel


def route_by_complexity(state: CharacterState) -> str:
    """
    Route selon la complexité de la requête.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        Nom du prochain nœud à exécuter
    """
    complexity = state["complexity_level"]
    
    # Ajout d'informations de debug
    state["processing_steps"].append(f"complexity_routing_{complexity}")
    
    # Routage selon la complexité
    if complexity == ComplexityLevel.SIMPLE:
        return "simple_response"
    elif complexity == ComplexityLevel.MEDIUM:
        return "assess_rag_need"
    else:  # ComplexityLevel.COMPLEX
        return "assess_rag_need"


def route_by_rag_need(state: CharacterState) -> str:
    """
    Route selon le besoin de recherche RAG.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        Nom du prochain nœud à exécuter
    """
    needs_rag = state["needs_rag_search"]
    
    # Ajout d'informations de debug
    state["processing_steps"].append(f"rag_routing_{'rag' if needs_rag else 'direct'}")
    
    if needs_rag:
        return "rag_search"
    else:
        return "generate_response"


def check_if_needs_memory_update(state: CharacterState) -> str:
    """
    Détermine si une mise à jour de la mémoire est nécessaire.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        Nom du prochain nœud à exécuter
    """
    # Critères pour mise à jour mémoire
    message_length = len(state["user_message"].split())
    has_emotional_content = state["message_intent"] == "emotional"
    has_game_actions = bool(state["planned_actions"])
    
    needs_memory_update = (
        message_length > 5 or 
        has_emotional_content or 
        has_game_actions or
        state["complexity_level"] == ComplexityLevel.COMPLEX
    )
    
    state["processing_steps"].append(f"memory_routing_{'update' if needs_memory_update else 'skip'}")
    
    if needs_memory_update:
        return "memory_update"
    else:
        return "finalize"