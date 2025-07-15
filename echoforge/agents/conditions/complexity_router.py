"""Routeurs conditionnels pour les graphes LangGraph."""

from ..state.character_state import CharacterState
from echoforge.utils.config import get_config

config = get_config()

def route_by_complexity(state: CharacterState) -> str:
    """
    Route selon la complexité de la requête.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        Nom du prochain nœud à exécuter
    """
    # complexity = state["complexity_level"]
    
    # # Ajout d'informations de debug
    # state["processing_steps"].append(f"complexity_routing_{complexity}")
    
    # # Routage selon la complexité
    # if complexity == ComplexityLevel.SIMPLE:
    #     return "simple_response"
    # elif complexity == ComplexityLevel.MEDIUM:
    #     return "assess_rag_need"
    # else:  # ComplexityLevel.COMPLEX
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

def check_if_needs_new_rag(state: CharacterState) -> str:
    """
    Route selon le besoin d'une nouvelle recherche RAG.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        Nom du prochain nœud à exécuter
    """
    needs_rag_retry = state["needs_rag_retry"]
    
    # Ajout d'informations de debug
    state["processing_steps"].append(f"rag_routing_{'retry' if needs_rag_retry else 'end'}")
    
    if needs_rag_retry:
        return "rag_retry"
    else:
        return "generate_response"
