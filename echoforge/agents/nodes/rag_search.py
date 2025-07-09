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
    def fn(llm_manager):
        state["processing_steps"].append("rag_search")
        query = state["rag_query"][-1]

        if state.get("needs_rag_retry"):
            new_query = _reformulate_query_with_llm(state, previous_query=query, llm_manager=llm_manager)
            if new_query:
                state["rag_query"].append(new_query)
                query = new_query
        
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
        state["rag_results"].extend(rag_results)
        
        # Debug info
        state["debug_info"]["rag_search"] = {
            "query": query,
            "results_count": len(rag_results),
            "top_relevance_score": rag_results[0]["relevance"] if rag_results else 0
        }
        
        return state
    return fn

def _reformulate_query_with_llm(state: CharacterState, previous_query: str, llm_manager) -> str:
    """
    Reformule une requête RAG plus efficace à partir du message utilisateur,
    de l’intention, et du contexte de recherche précédent.
    """
    user_msg = state.get("parsed_message") or state.get("user_message", "")
    intent = state.get("message_intent", "")
    character_name = state.get("character_name", "le personnage")
    rag_results = state.get("rag_results", [])
    
    previous_knowledge = "\n".join(
        f"- {r['content']}" for r in rag_results[:3]
    ) if rag_results else "Aucun résultat pertinent précédemment trouvé."

    prompt = f"""
Tu es un assistant expert en recherche de connaissances narratives pour un jeu de rôle.

Le personnage s'appelle {character_name}.

Le joueur a dit : "{user_msg}"
Intention détectée : {intent}
Ancienne requête utilisée : "{previous_query}"

Voici les résultats précédemment trouvés :
{previous_knowledge}

Tu dois générer une nouvelle requête optimisée, plus précise, qui aiderait le personnage à trouver des informations pertinentes.

Réponds uniquement par la requête reformulée, sans autre texte.
Si aucune reformulation pertinente n'est possible, réponds exactement : NONE
"""

    try:
        new_query = llm_manager.invoke(prompt).strip()
        if new_query.upper() == "NONE":
            return None
        return new_query
    except Exception as e:
        state["debug_info"]["query_reformulation_error"] = str(e)
        return None