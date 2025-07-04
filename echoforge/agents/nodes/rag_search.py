"""Nœud de recherche RAG."""

from typing import List, Dict, Any
from ..state.character_state import CharacterState


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
    
    # Simulation d'une recherche RAG (à remplacer par ton système RAG)
    rag_results = _simulate_rag_search(query, character_name)
    
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


def _simulate_rag_search(query: str, character_name: str) -> List[Dict[str, Any]]:
    """
    Simulation d'une recherche RAG.
    
    TODO: Remplacer par l'intégration avec ton système RAG réel
    """
    
    # Base de connaissances simulée par personnage
    knowledge_base = {
        "fathira": [
            {
                "content": "Fathira est maire de l'île depuis 10 ans. Elle connaît tous les secrets de l'île et possède un trésor ancestral.",
                "metadata": {"type": "background", "importance": "high"},
                "relevance": 0.9
            },
            {
                "content": "L'île a une histoire mystérieuse. Elle était autrefois habitée par une ancienne civilisation qui a laissé des trésors cachés.",
                "metadata": {"type": "history", "importance": "medium"},
                "relevance": 0.8
            },
            {
                "content": "Fathira entretient de bonnes relations avec tous les habitants de l'île. Elle est respectée pour sa sagesse.",
                "metadata": {"type": "relationships", "importance": "medium"},
                "relevance": 0.7
            }
        ],
        "claude": [
            {
                "content": "Claude est forgeron depuis 20 ans. Il peut réparer n'importe quoi mais aime négocier pour ses services.",
                "metadata": {"type": "background", "importance": "high"},
                "relevance": 0.9
            },
            {
                "content": "L'atelier de Claude contient des outils anciens et des secrets de métallurgie transmis de génération en génération.",
                "metadata": {"type": "knowledge", "importance": "medium"},
                "relevance": 0.8
            }
        ],
        "azzedine": [
            {
                "content": "Azzedine est un styliste talentueux qui vend des tissus rares. Il est perfectionniste et exigeant sur la qualité.",
                "metadata": {"type": "background", "importance": "high"},
                "relevance": 0.9
            }
        ],
        "roberte": [
            {
                "content": "Roberte est une cuisinière réputée. Elle déteste être dérangée pendant son travail mais offre volontiers des cookies en pause.",
                "metadata": {"type": "background", "importance": "high"},
                "relevance": 0.9
            }
        ]
    }
    
    # Recherche basique par mots-clés
    query_lower = query.lower()
    results = []
    
    # Recherche dans la base du personnage spécifique
    if character_name.lower() in knowledge_base:
        character_knowledge = knowledge_base[character_name.lower()]
        
        for item in character_knowledge:
            # Score de pertinence basique
            content_lower = item["content"].lower()
            relevance_score = 0
            
            # Compte les mots de la requête présents dans le contenu
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    relevance_score += 0.1
            
            # Ajoute le score de base
            relevance_score += item["relevance"] * 0.5
            
            if relevance_score > 0.3:  # Seuil de pertinence
                results.append({
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "relevance": min(relevance_score, 1.0),
                    "source": f"{character_name}_knowledge"
                })
    
    # Recherche dans les connaissances générales de l'île
    general_knowledge = [
        {
            "content": "L'île mystérieuse est un lieu magique où vivent quatre habitants principaux : Fathira la maire, Claude le forgeron, Azzedine le styliste et Roberte la cuisinière.",
            "metadata": {"type": "general", "importance": "high"},
            "relevance": 0.6
        },
        {
            "content": "Les habitants de l'île ont développé un système d'échange basé sur l'or, les cookies et les services.",
            "metadata": {"type": "economy", "importance": "medium"}, 
            "relevance": 0.5
        }
    ]
    
    for item in general_knowledge:
        content_lower = item["content"].lower()
        relevance_score = 0
        
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2 and word in content_lower:
                relevance_score += 0.1
        
        relevance_score += item["relevance"] * 0.3
        
        if relevance_score > 0.2:
            results.append({
                "content": item["content"],
                "metadata": item["metadata"],
                "relevance": min(relevance_score, 1.0),
                "source": "general_knowledge"
            })
    
    # Trie par pertinence décroissante
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Limite à 3 résultats maximum
    return results[:3]