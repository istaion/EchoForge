"""Nœud de recherche RAG."""

from typing import List, Dict, Any
from ..state.character_state import CharacterState
from langsmith import traceable
from echoforge.core import EchoForgeRAG
from echoforge.utils.config import get_config

config = get_config()
@traceable
def perform_rag_search(state: CharacterState) -> CharacterState:
    """
    Effectue une recherche RAG basée sur la requête déterminée.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec les résultats RAG
    """
    state["processing_steps"].append("rag_search")
    
    query = state["rag_query"]
    character_name = state["character_name"]
    try:
        rag_system = EchoForgeRAG(
                data_path=str(config.data_path),
                vector_store_path=str(config.vector_store_path),
                embedding_model=config.embedding_model,
                llm_model=config.llm_model
            )
        results = []

        # Recherche dans les connaissances du monde
        world_context = rag_system.retrieve_world_context(query, top_k=config.top_k_world)
        for i, content in enumerate(world_context):
            results.append({
                "content": content,
                "metadata": {"type": "world", "importance": "medium"},
                "relevance": max(0.8 - i * 0.1, 0.3),  # Score décroissant
                "source": "world_knowledge"
            })
        
        # Recherche dans les connaissances du personnage
        character_context = rag_system.retrieve_character_context(
            query, character_name.lower(), top_k=config.top_k_character
        )
        for i, content in enumerate(character_context):
            results.append({
                "content": content,
                "metadata": {"type": "character", "importance": "high"},
                "relevance": max(0.9 - i * 0.1, 0.4),  # Score plus élevé pour le personnage
                "source": f"{character_name}_knowledge"
            })
        
        # Trie par pertinence décroissante
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
    except ImportError as e:
        print("⚠️ EchoForgeRAG non disponible, utilisation de la simulation")
        return state
    
    except Exception as e:
        print(f"⚠️ Erreur lors de la recherche RAG: {e}")
        return state
    
    # Limite à 5 résultats maximum
    rag_results = results[:5]
    
    # Mise à jour de l'état
    state["rag_results"] = rag_results
    state["relevant_knowledge"] = [result["content"] for result in rag_results]
    
    # Debug info
    state["debug_info"]["rag_search"] = {
        "query": query,
        "results_count": len(rag_results),
        "top_relevance_score": rag_results[0]["relevance"] if rag_results else 0
    }
    
    return state

