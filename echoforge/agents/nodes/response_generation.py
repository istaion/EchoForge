import random
from typing import Dict, Any, List
from ..state.character_state import CharacterState
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()

@traceable
def generate_simple_response(state: CharacterState) -> CharacterState:
    """
    Génère une réponse simple basée sur la personnalité, sans RAG.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la réponse générée
    """
    state["processing_steps"].append("simple_response_generation")
    
    character_name = state["character_name"]
    personality = state["character_data"]["personality"]
    intent = state["message_intent"]
    emotion = state["character_data"]["current_emotion"]
    
    # Génération de réponse simple selon l'intention
    response = _generate_personality_response(
        character_name, personality, intent, emotion, state["user_message"]
    )
    
    state["response"] = response
    
    # Debug info
    state["debug_info"]["response_generation"] = {
        "type": "simple",
        "intent_used": intent,
        "emotion_used": emotion,
        "personality_traits_count": len(personality)
    }
    
    return state

@traceable
def generate_response(state: CharacterState) -> CharacterState:
    """
    Génère une réponse complexe avec ou sans contexte RAG via LLM.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la réponse générée
    """
    state["processing_steps"].append("complex_response_generation")
    
    # Génération via LLM avec contexte
    response = _generate_llm_response(state)
    
    state["response"] = response["content"]
    
    # Debug info
    state["debug_info"]["response_generation"] = {
        "type": "llm_generated",
        "rag_results_used": len(state["rag_results"]),
        "context_elements": len(state["relevant_knowledge"]),
        "response_length": len(response["content"]),
        "method": response.get("method", "llm_call"),
        "success": response.get("success", True)
    }
    
    return state

@traceable
def _generate_llm_response(state: CharacterState) -> Dict[str, Any]:
    """
    Génère une réponse via LLM avec tout le contexte disponible.
    
    Args:
        state: État du personnage avec contexte
        
    Returns:
        Dict avec la réponse et métadonnées
    """
    try:
        # Import du système LLM
        from echoforge.core import EchoForgeRAG
        from echoforge.utils.config import get_config
        
        # Récupération de la configuration et du LLM
        config = get_config()
        rag_system = EchoForgeRAG(
            data_path=str(config.data_path),
            vector_store_path=str(config.vector_store_path),
            embedding_model=config.embedding_model,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model
        )
        
        # Construction du prompt complet
        response_prompt = _build_comprehensive_prompt(state)
        
        # Appel au LLM
        llm_response = rag_system.llm.invoke(response_prompt)
        
        return {
            "content": llm_response.strip(),
            "method": "llm_call",
            "success": True,
            "prompt_length": len(response_prompt)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la génération LLM: {e}")
        # Fallback sur génération simple
        fallback_response = _generate_fallback_response(state)
        return {
            "content": fallback_response,
            "method": "fallback",
            "success": False,
            "error": str(e)
        }


def _build_comprehensive_prompt(state: CharacterState) -> str:
    """Construit un prompt complet avec tout le contexte disponible."""
    
    # Récupération des données
    character_name = state["character_name"]
    personality = state["character_data"].get("personality")
    emotion = state["character_data"].get("current_emotion")
    user_relation = state["character_data"].get("relation")
    user_message = state["user_message"]
    rag_results = state["rag_results"]
    conversation_history = state.get("conversation_history", [])
    all_triggers = state["character_data"].get("triggers", {}).get("input", {})
    activated = state.get("activated_input_triggers", []) or []
    refused = [t for t in all_triggers.keys() if t not in activated]
    
    # Format des intentions activées
    def format_trigger_list(trigger_names):
        return "\n".join([
            f"- {trigger}: {all_triggers[trigger].get('trigger')} → effet attendu : {all_triggers[trigger].get('effect')}"
            for trigger in trigger_names if trigger in all_triggers
        ]) or "Aucune"
    
    # Construction des sections du prompt
    
    # 1. Identité du personnage
    identity_section = f"""Tu es {character_name}.
PERSONNALITÉ: {_format_personality(personality)}
ÉMOTION ACTUELLE: {emotion}
NIVEAU RELATION AVEC LE PERSONNAGE JOUEUR (entre -10 et 10) : {user_relation}"""

    # 2. Contexte RAG si disponible
    context_section = ""
    if rag_results:
        context_items = []
        for result in rag_results[:3]:  # Limite à 3 résultats les plus pertinents
            context_items.append(f"- {result['content']} (pertinence: {result['relevance']:.2f})")
        
        context_section = f"""
CONNAISSANCES PERTINENTES:
{chr(10).join(context_items)}"""

    # 3. Historique de conversation récent
    history_section = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # 3 derniers échanges
        history_items = []
        for exchange in recent_history:
            if isinstance(exchange, dict):
                if "user" in exchange and "assistant" in exchange:
                    history_items.append(f"Utilisateur: {exchange['user']}")
                    history_items.append(f"Toi: {exchange['assistant']}")
        
        if history_items:
            history_section = f"""
HISTORIQUE RÉCENT:
{chr(10).join(history_items)}"""

    # === 4. Section sur les intentions détectées ===
    intent_section = f"""
INTENTIONS DÉTECTÉES DANS LE MESSAGE DU JOUEUR :
- Intentions activées :
{format_trigger_list(activated)}

- Intentions non activées (refusées ou ignorées) :
{format_trigger_list(refused)}
"""

    # 5. Instructions spécifiques
    instructions_section = f"""
MESSAGE ACTUEL DE L'UTILISATEUR: "{user_message}"

INSTRUCTIONS:
1. Reste parfaitement en personnage comme {character_name}
2. Utilise ta personnalité et ton émotion actuelle ({emotion})
3. Si tu as des connaissances pertinentes ci-dessus, intègre-les naturellement
4. Tiens compte de l'historique de conversation pour la cohérence
5. Si tu fais des actions physiques, mets-les entre *astérisques*
6. Réponds en français de manière authentique à ton personnage

RÉPONSE:"""

    # === Assemblage final ===
    full_prompt = f"""{identity_section}{context_section}{history_section}
{intent_section}
{instructions_section}"""
    
    return full_prompt


def _format_personality(personality: Dict[str, Any]) -> str:
    """Formate les traits de personnalité pour le prompt."""
    if not personality:
        return "Personnalité standard"
    
    if isinstance(personality, dict):
        traits = []
        for key, value in personality.items():
            if isinstance(value, (int, float)):
                if value > 0.7:
                    traits.append(f"{key} élevé")
                elif value < 0.3:
                    traits.append(f"{key} faible")
            else:
                traits.append(f"{key}: {value}")
        return ", ".join(traits) if traits else "Équilibré"
    
    return str(personality)


def _generate_fallback_response(state: CharacterState) -> str:
    """Génère une réponse de fallback si le LLM échoue."""
    
    character_name = state["character_name"]
    intent = state["message_intent"]
    emotion = state["current_emotion"]
    
    # Templates de fallback par intention
    fallback_templates = {
        "question": [
            f"C'est une question intéressante... Laissez-moi réfléchir.",
            f"Hmm, votre question me fait réfléchir.",
            f"Interessant que vous me demandiez cela."
        ],
        "greeting": [
            f"Bonjour ! Comment puis-je vous aider ?",
            f"Salutations ! Que puis-je faire pour vous ?",
            f"Bienvenue ! En quoi puis-je vous être utile ?"
        ],
        "request": [
            f"Je vais voir ce que je peux faire pour vous.",
            f"Votre demande mérite considération.",
            f"Laissez-moi réfléchir à votre requête."
        ],
        "general": [
            f"Je vous écoute attentivement.",
            f"Continuez, vous avez mon attention.",
            f"Intéressant, dites-moi en plus."
        ]
    }
    
    # Sélection d'un template approprié
    templates = fallback_templates.get(intent, fallback_templates["general"])
    response = random.choice(templates)
    
    # Ajustement selon l'émotion
    if emotion == "sad":
        response += " *soupire*"
    elif emotion == "happy":
        response += " 😊"
    elif emotion == "angry":
        response = response.replace("!", ".")
    
    return response


def _generate_personality_response(
    character_name: str, 
    personality: Dict[str, Any], 
    intent: str, 
    emotion: str, 
    user_message: str
) -> str:
    """Génère une réponse simple basée uniquement sur la personnalité."""
    
    # Templates de réponses par personnage et intention
    response_templates = {
        "martine": {
            "greeting": [
                "Bonjour, citoyen ! Comment puis-je vous aider aujourd'hui ?",
                "Salutations ! En tant que maire, je suis à votre service.",
                "Bienvenue ! Que puis-je faire pour notre communauté ?"
            ],
            "farewell": [
                "Au revoir, citoyen. Que votre journée soit prospère !",
                "Prenez soin de vous et de notre belle île !",
                "À bientôt ! N'hésitez pas à revenir me voir."
            ],
            "general": [
                "En tant que maire, je veille sur tous les habitants de notre île.",
                "Notre communauté est ma priorité absolue, citoyen.",
                "Parlons de ce qui vous préoccupe."
            ]
        },
        "claude": {
            "greeting": [
                "Salut ! Mon marteau et moi sommes à votre service.",
                "Bonjour ! Besoin de quelque chose de réparé ?",
                "Hey ! L'atelier est ouvert, que puis-je forger pour vous ?"
            ],
            "farewell": [
                "À plus ! Si ça casse, vous savez où me trouver !",
                "Prenez soin de vos outils !",
                "Au revoir ! Que le métal soit avec vous !"
            ],
            "general": [
                "Je peux réparer à peu près n'importe quoi, mais ça se négocie.",
                "Mon atelier a vu passer bien des objets étranges...",
                "Le métal ne ment jamais, contrairement aux gens."
            ]
        },
        "azzedine": {
            "greeting": [
                "Bonjour ! Admirez-vous ma dernière création ?",
                "Salut ! L'art textile vous intéresse-t-il ?",
                "Bienvenue dans mon atelier de beauté !"
            ],
            "farewell": [
                "Au revoir ! Que la beauté vous accompagne !",
                "Partez avec style !",
                "À bientôt ! Revenez quand vous aurez développé votre goût esthétique."
            ],
            "general": [
                "La beauté est dans les détails, et je suis un perfectionniste.",
                "Mes tissus sont d'une qualité incomparable sur cette île.",
                "L'art véritable demande de la patience et de l'exigence."
            ]
        },
        "roberte": {
            "greeting": [
                "Bonjour ! J'espère que vous avez faim !",
                "Salut ! La cuisine sent bon aujourd'hui, n'est-ce pas ?",
                "Bienvenue ! Venez-vous pour mes fameux cookies ?"
            ],
            "farewell": [
                "Au revoir ! Revenez quand vous aurez faim !",
                "À bientôt ! Mes cookies vous attendront !",
                "Prenez soin de vous et mangez bien !"
            ],
            "general": [
                "Ma cuisine est ma fierté, ne me dérangez pas pendant que je travaille !",
                "Un bon repas réchauffe le cœur et l'âme.",
                "Mes cookies sont les meilleurs de l'île, mais il faut les mériter !"
            ]
        }
    }
    
    # Sélection du template approprié
    character_templates = response_templates.get(character_name.lower(), response_templates["martine"])
    intent_templates = character_templates.get(intent, character_templates["general"])
    
    # Sélection aléatoire d'une réponse
    response = random.choice(intent_templates)
    
    return response
